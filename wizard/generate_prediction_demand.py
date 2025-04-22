# -*- coding: utf-8 -*-

from odoo import models, fields, api, _
from odoo.exceptions import UserError, ValidationError
import logging
import base64
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import time
from dateutil.relativedelta import relativedelta
import calendar
import traceback
from odoo import tools

_logger = logging.getLogger(__name__)


class GenerateDemandPredictionWizard(models.TransientModel):
    _name = 'generate.demand.prediction.wizard'
    _description = 'Generate Demand Prediction Wizard'

    forecast_name = fields.Char('Forecast Name', required=True, help="A name to identify this demand forecast batch")
    forecast_description = fields.Text('Description', help="Additional details about this demand forecast")
    forecast_start_date = fields.Date('Forecast Start Date', required=True, default=fields.Date.today)
    forecast_days = fields.Integer('Number of Days', required=True, default=30)
    product_selection = fields.Selection([
        ('all', 'All Products'),
        ('specific', 'Specific Products'),
        ('category', 'Product Category'),
    ], string='Product Selection', default='all', required=True)
    product_ids = fields.Many2many('product.product', string='Products')
    product_category_id = fields.Many2one('product.category', string='Product Category')
    model_id = fields.Many2one('demand.prediction.model', string='Model', required=True)
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse', required=True)
    prediction_period = fields.Selection([
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('quarterly', 'Quarterly'),
        ('yearly', 'Yearly')
    ], string='Prediction Period', default='daily', required=True)
    include_stockouts = fields.Boolean('Include Stockout Analysis', default=True,
                                       help="Analyze historical stockouts to estimate true demand")
    seasonality_factor = fields.Boolean('Apply Seasonality', default=True,
                                        help="Apply seasonal adjustments to demand forecasts")
    safety_stock_days = fields.Integer('Safety Stock Days', default=7,
                                       help="Number of days of safety stock to include in demand calculations")
    debug_mode = fields.Boolean('Debug Mode', default=True, help="Enable detailed logging for debugging")

    @api.constrains('forecast_days')
    def _check_forecast_days(self):
        for record in self:
            if record.forecast_days <= 0:
                raise ValidationError(_("Number of days must be greater than zero."))

    @api.constrains('product_ids', 'product_selection', 'product_category_id')
    def _check_products(self):
        for record in self:
            if record.product_selection == 'specific' and not record.product_ids:
                raise ValidationError(_("Please select at least one product."))
            if record.product_selection == 'category' and not record.product_category_id:
                raise ValidationError(_("Please select a product category."))

    def action_generate_demand_predictions(self):
        """Generate demand predictions using the selected model"""
        self.ensure_one()

        start_time = time.time()

        try:
            # Check model type to determine which files are required
            model_type = self.model_id.model_type
            _logger.info(f"Starting demand prediction generation with model type: {model_type}")

            # Load the model file (required for all model types)
            model_data = self.model_id.model_file
            if not model_data:
                raise UserError(_("Model file is missing. Please make sure the model is properly imported."))

            # Decode base64 data and load model
            model_binary = base64.b64decode(model_data)
            model = pickle.loads(model_binary)

            _logger.info(f"Model loaded successfully: {type(model).__name__}")

            # For non-Prophet models, load scaler and features
            if model_type != 'prophet':
                scaler_data = self.model_id.scaler_file
                features_data = self.model_id.features_file

                if not scaler_data or not features_data:
                    raise UserError(
                        _("Scaler or features files are missing. Please make sure the model is properly imported."))

                # Decode base64 data and load scaler and features
                scaler_binary = base64.b64decode(scaler_data)
                features_binary = base64.b64decode(features_data)

                scaler = pickle.loads(scaler_binary)
                features = pickle.loads(features_binary)

                _logger.info(f"Scaler loaded: {type(scaler).__name__}")
                _logger.info(f"Features loaded: {len(features)} features")
            else:
                # For Prophet models, set scaler and features to None
                scaler = None
                features = None

            # Get products to forecast
            if self.product_selection == 'all':
                products = self.env['product.product'].search([('active', '=', True)])
            elif self.product_selection == 'category':
                products = self.env['product.product'].search([
                    ('active', '=', True),
                    ('categ_id', '=', self.product_category_id.id)
                ])
            else:
                products = self.product_ids

            _logger.info(f"Found {len(products)} products for demand forecasting")

            # Get historical sales data
            sales_data = self._get_historical_sales_data(products)

            # Get stockout data if enabled
            stockout_data = None
            if self.include_stockouts:
                stockout_data = self._get_historical_stockout_data(products)
                _logger.info(
                    f"Historical stockout data loaded: {len(stockout_data) if stockout_data is not None else 0} records")

            _logger.info(f"Historical sales data loaded: {len(sales_data)} records")
            if not sales_data.empty:
                _logger.info(f"Products with data: {sales_data['product_id'].nunique()}")
            else:
                _logger.warning("No historical sales data found!")

            # Create a demand prediction batch first
            end_date = self._calculate_end_date()
            batch = self.env['demand.prediction.batch'].create({
                'name': self.forecast_name,
                'description': self.forecast_description,
                'date': fields.Date.today(),
                'user_id': self.env.user.id,
                'prediction_period': self.prediction_period,
                'model_id': self.model_id.id,
                'warehouse_id': self.warehouse_id.id,
                'start_date': self.forecast_start_date,
                'end_date': end_date,
                'include_stockouts': self.include_stockouts,
                'seasonality_factor': self.seasonality_factor,
                'safety_stock_days': self.safety_stock_days,
                'state': 'draft',
            })

            _logger.info(f"Created demand prediction batch: {batch.id}")

            # Generate predictions for each product
            predictions = []
            skipped_products = 0

            for product in products:
                try:
                    _logger.info(f"Processing product demand: {product.name} (ID: {product.id})")

                    # Filter sales data for this product
                    product_sales = sales_data[sales_data['product_id'] == product.id]

                    # Get stockout data for this product if available
                    product_stockouts = None
                    if stockout_data is not None:
                        product_stockouts = stockout_data[stockout_data['product_id'] == product.id]

                    # If no historical data, set prediction to 0
                    if product_sales.empty:
                        _logger.warning(
                            f"No historical data for product {product.name}. Setting demand prediction to 0.")
                        zero_predictions = self._generate_zero_demand_prediction(product, batch.id)
                        predictions.extend(zero_predictions)
                        _logger.info(f"Generated zero demand prediction for product {product.name}")
                        continue

                    # Combine sales and stockout data to estimate true demand
                    product_demand = self._estimate_true_demand(product_sales, product_stockouts, product)

                    # For Prophet models, we need a special approach
                    if model_type == 'prophet':
                        product_predictions = self._predict_demand_with_prophet(
                            product,
                            product_demand,
                            model,
                            batch.id
                        )
                    else:
                        # Get the number of periods to forecast
                        periods_to_forecast = self._get_periods_to_forecast()

                        # Generate future dates
                        future_dates = self._generate_future_dates(periods_to_forecast)

                        # Generate predictions using ML model
                        product_predictions = self._predict_demand_with_ml_model(
                            product,
                            product_demand,
                            model,
                            scaler,
                            features,
                            future_dates,
                            batch.id
                        )

                    if product_predictions:
                        predictions.extend(product_predictions)
                        _logger.info(
                            f"Generated {len(product_predictions)} demand predictions for product {product.name}")
                    else:
                        # If no predictions were generated, set to zero
                        _logger.warning(f"No demand predictions generated for product {product.name}. Setting to zero.")
                        zero_predictions = self._generate_zero_demand_prediction(product, batch.id)
                        predictions.extend(zero_predictions)
                        _logger.info(f"Generated zero demand prediction for product {product.name}")

                except Exception as e:
                    skipped_products += 1
                    _logger.error(f"Error generating demand predictions for product {product.name}: {str(e)}")
                    _logger.error(traceback.format_exc())

                    # Use zero prediction on error
                    try:
                        zero_predictions = self._generate_zero_demand_prediction(product, batch.id)
                        predictions.extend(zero_predictions)
                        _logger.info(f"Generated zero demand prediction after error for product {product.name}")
                    except Exception as e2:
                        _logger.error(f"Failed to create zero demand prediction for {product.name}: {str(e2)}")

            _logger.info(
                f"Generated a total of {len(predictions)} demand predictions. Skipped {skipped_products} products.")

            # Create demand prediction records
            self._create_demand_prediction_records(predictions)

            # Log the operation
            execution_time = time.time() - start_time
            self.env['demand.prediction.log'].log_operation(
                'predict',
                'success',
                f"Generated {len(predictions)} demand predictions using model {self.model_id.name} with name '{self.forecast_name}'",
                execution_time,
                batch.id
            )

            # Show success message and open the batch
            return {
                'name': _('Demand Prediction Batch'),
                'type': 'ir.actions.act_window',
                'res_model': 'demand.prediction.batch',
                'view_mode': 'form',
                'res_id': batch.id,
                'context': {'form_view_initial_mode': 'edit'},
            }

        except Exception as e:
            # Log the error
            execution_time = time.time() - start_time
            _logger.error(f"Error generating demand predictions: {str(e)}")
            _logger.error(traceback.format_exc())

            self.env['demand.prediction.log'].log_operation(
                'predict',
                'failed',
                f"Error generating demand predictions: {str(e)}",
                execution_time
            )

            raise UserError(_(f"Error generating demand predictions: {str(e)}"))

    def _estimate_true_demand(self, sales_data, stockout_data, product):
        """
        Estime la vraie demande avec la logique :
        - Si stock == ventes ➜ demande = ventes * 1.3
        - Sinon ➜ demande = stock - ventes
        """
        _logger.info(f"Calcul de la vraie demande pour {product.name} selon la logique personnalisée")

        # Calcul des ventes totales
        total_sales = sales_data['quantity'].sum() if not sales_data.empty else 0

        # Calcul du stock actuel du produit dans l'entrepôt
        stock = product.with_context(warehouse=self.warehouse_id.id).qty_available

        # Application de ta formule
        if abs(stock - total_sales) < 0.01:  # tolérance pour float
            true_quantity = total_sales * 1.3
        else:
            true_quantity = max(0.0, stock - total_sales)  # évite valeurs négatives

        # Préparer DataFrame minimal pour la suite
        demand_data = sales_data.copy()
        demand_data['true_quantity'] = true_quantity
        demand_data['true_amount'] = true_quantity * product.list_price  # tu peux adapter

        # Ajouter une seule ligne si vide
        if demand_data.empty:
            demand_data = pd.DataFrame([{
                'date': self.forecast_start_date,
                'product_id': product.id,
                'quantity': 0,
                'price': product.list_price,
                'true_quantity': true_quantity,
                'true_amount': true_quantity * product.list_price
            }])

        # Ajouter les ajustements de saison si activé
        if self.seasonality_factor:
            demand_data = self._apply_seasonality_adjustments(demand_data)

        return demand_data

    def _apply_seasonality_adjustments(self, demand_data):
        """
        Apply seasonality adjustments to demand data
        """
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(demand_data['date']):
            demand_data['date'] = pd.to_datetime(demand_data['date'])

        # Extract month and day of week
        demand_data['month'] = demand_data['date'].dt.month
        demand_data['dayofweek'] = demand_data['date'].dt.dayofweek

        # Calculate monthly seasonality factors
        monthly_factors = {
            1: 1.1,  # January (post-holiday)
            2: 0.9,  # February
            3: 1.0,  # March
            4: 1.0,  # April
            5: 1.1,  # May
            6: 1.2,  # June (summer)
            7: 1.2,  # July (summer)
            8: 1.1,  # August (back to school)
            9: 1.0,  # September
            10: 1.0,  # October
            11: 1.2,  # November (holiday season)
            12: 1.3,  # December (holiday season)
        }

        # Calculate day of week seasonality factors
        dayofweek_factors = {
            0: 1.0,  # Monday
            1: 1.0,  # Tuesday
            2: 1.0,  # Wednesday
            3: 1.1,  # Thursday
            4: 1.2,  # Friday
            5: 1.3,  # Saturday
            6: 1.1,  # Sunday
        }

        # Apply seasonality factors
        demand_data['month_factor'] = demand_data['month'].map(monthly_factors)
        demand_data['dayofweek_factor'] = demand_data['dayofweek'].map(dayofweek_factors)

        # Apply combined seasonality factor
        demand_data['seasonality_factor'] = demand_data['month_factor'] * demand_data['dayofweek_factor']

        # Adjust true quantity and amount with seasonality
        demand_data['true_quantity'] = demand_data['true_quantity'] * demand_data['seasonality_factor']
        demand_data['true_amount'] = demand_data['true_amount'] * demand_data['seasonality_factor']

        return demand_data

    def _predict_demand_with_prophet(self, product, product_demand, prophet_model, batch_id):
        """Generate demand predictions using Prophet model"""
        _logger.info(f"Using Prophet approach for product demand {product.name}")

        try:
            # Prepare data for Prophet
            df = product_demand.groupby('date')['true_amount'].sum().reset_index()
            df.columns = ['ds', 'y']

            _logger.info(f"Prophet input data prepared with {len(df)} rows")

            # Get future dates based on prediction period
            periods_to_forecast = self._get_periods_to_forecast()
            future_dates = self._generate_future_dates(periods_to_forecast)

            # Create future dataframe for Prophet
            future_df = pd.DataFrame({'ds': future_dates})

            # Make predictions
            try:
                forecast = prophet_model.predict(future_df)
                _logger.info(f"Prophet demand prediction successful with {len(forecast)} rows")
            except Exception as prophet_error:
                _logger.error(f"Prophet demand prediction failed: {str(prophet_error)}")
                _logger.error(traceback.format_exc())

                # If Prophet prediction fails, return zero prediction
                return self._generate_zero_demand_prediction(product, batch_id)

            # Create prediction records
            result = []
            for i, row in forecast.iterrows():
                date = row['ds'].date()

                # Skip dates in the past
                if date < fields.Date.today():
                    continue

                # Get the prediction value (yhat)
                prediction_value = max(1.0, row['yhat'])  # Ensure positive value

                # Add product-specific factor to ensure different predictions
                product_factor = 0.8 + ((product.id % 100) / 100) * 0.4  # Factor between 0.8 and 1.2
                prediction_value = prediction_value * product_factor

                # Add safety stock based on safety_stock_days
                if self.safety_stock_days > 0:
                    # Calculate daily average from prediction
                    daily_avg = prediction_value / 30  # Assuming monthly average
                    safety_stock = daily_avg * self.safety_stock_days
                    prediction_value += safety_stock

                # Calculate start and end dates for the period
                start_date, end_date = self._calculate_period_dates(date)

                result.append({
                    'date': date,
                    'product_id': product.id,
                    'warehouse_id': self.warehouse_id.id,
                    'prediction_value': prediction_value,
                    'prediction_period': self.prediction_period,
                    'model_id': self.model_id.id,
                    'generator_id': self.env.user.id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'name': self.forecast_name,
                    'description': f"{self.forecast_description} (Prophet demand prediction)",
                    'batch_id': batch_id,
                    'includes_safety_stock': self.safety_stock_days > 0,
                    'safety_stock_days': self.safety_stock_days,
                })

            _logger.info(f"Generated {len(result)} Prophet demand predictions for product {product.name}")
            return result

        except Exception as e:
            _logger.error(f"Error in _predict_demand_with_prophet for {product.name}: {str(e)}")
            _logger.error(traceback.format_exc())
            return self._generate_zero_demand_prediction(product, batch_id)

    def _generate_zero_demand_prediction(self, product, batch_id):
        """Generate a zero demand prediction for a product with no historical data"""
        _logger.info(f"Generating zero demand prediction for product {product.name}")

        # Get the date
        date = self.forecast_start_date

        # Calculate start and end dates
        start_date, end_date = self._calculate_period_dates(date)

        return [{
            'date': date,
            'product_id': product.id,
            'warehouse_id': self.warehouse_id.id,
            'prediction_value': 0.0,  # Set prediction to zero
            'prediction_period': self.prediction_period,
            'model_id': self.model_id.id,
            'generator_id': self.env.user.id,
            'start_date': start_date,
            'end_date': end_date,
            'name': self.forecast_name,
            'description': f"{self.forecast_description} (Zero demand prediction - no historical data)",
            'batch_id': batch_id,
            'includes_safety_stock': False,
            'safety_stock_days': 0,
        }]

    def _predict_demand_with_ml_model(self, product, product_demand, model, scaler, features, future_dates, batch_id):
        """Generate demand predictions using ML model (XGBoost, Random Forest, etc.)"""
        try:
            _logger.info(f"Using ML model for product demand {product.name}")

            # Calculate product-specific statistics for features
            product_stats = {
                'avg_price': product_demand['price'].mean() if not product_demand.empty else 0,
                'avg_quantity': product_demand['true_quantity'].mean() if not product_demand.empty else 0,
                'avg_amount': product_demand['true_amount'].mean() if not product_demand.empty else 0,
                'total_quantity': product_demand['true_quantity'].sum() if not product_demand.empty else 0,
                'total_amount': product_demand['true_amount'].sum() if not product_demand.empty else 0,
                'demand_count': len(product_demand),
                'unique_customers': product_demand['partner_id'].nunique() if not product_demand.empty else 0,
            }

            # Add lead time as a feature if available
            lead_time = 1  # Default
            if product.seller_ids:
                lead_time = product.seller_ids[0].delay
            product_stats['lead_time'] = lead_time

            # Create features for prediction
            future_features = []
            for date in future_dates:
                # Skip dates in the past
                if date < fields.Date.today():
                    continue

                # Create time-based features
                features_dict = {
                    'dayofweek': date.weekday(),
                    'month': date.month,
                    'year': date.year,
                    'dayofyear': date.timetuple().tm_yday,
                    'dayofmonth': date.day,
                    'quarter': (date.month - 1) // 3 + 1,
                    'is_weekend': 1 if date.weekday() >= 5 else 0,
                }

                # Add product-specific features
                features_dict['product_id'] = product.id
                features_dict['lead_time'] = lead_time

                # Add seasonality features if enabled
                if self.seasonality_factor:
                    # Monthly seasonality
                    monthly_factors = {
                        1: 1.1, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.1, 6: 1.2,
                        7: 1.2, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.2, 12: 1.3
                    }
                    features_dict['month_factor'] = monthly_factors.get(date.month, 1.0)

                    # Day of week seasonality
                    dayofweek_factors = {
                        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.1
                    }
                    features_dict['dayofweek_factor'] = dayofweek_factors.get(date.weekday(), 1.0)

                    # Combined seasonality
                    features_dict['seasonality_factor'] = features_dict['month_factor'] * features_dict[
                        'dayofweek_factor']

                # Add product statistics as features
                for stat_name, stat_value in product_stats.items():
                    features_dict[f'product_{stat_name}'] = stat_value

                future_features.append(features_dict)

            # Convert to DataFrame
            future_df = pd.DataFrame(future_features)

            _logger.info(f"Created feature dataframe with {len(future_df)} rows and {len(future_df.columns)} columns")

            # Ensure all required features are present
            missing_features = [f for f in features if f not in future_df.columns]
            if missing_features:
                _logger.warning(f"Missing features: {missing_features}. Adding with default values.")
                for feature in missing_features:
                    future_df[feature] = 0

            # Select only the features used by the model
            try:
                X_future = future_df[features].copy()
                _logger.info(f"Selected {len(features)} features for demand prediction")
            except KeyError as ke:
                _logger.error(f"KeyError when selecting features: {str(ke)}")
                _logger.error(f"Available columns: {future_df.columns.tolist()}")
                _logger.error(f"Required features: {features}")
                return self._generate_zero_demand_prediction(product, batch_id)

            # Scale features
            try:
                X_future_scaled = scaler.transform(X_future)
                _logger.info("Features scaled successfully")
            except Exception as scale_error:
                _logger.error(f"Error scaling features: {str(scale_error)}")
                return self._generate_zero_demand_prediction(product, batch_id)

            # Make predictions
            try:
                predictions = model.predict(X_future_scaled)
                _logger.info(f"Model demand prediction successful with {len(predictions)} values")
            except Exception as pred_error:
                _logger.error(f"Error making demand predictions: {str(pred_error)}")
                return self._generate_zero_demand_prediction(product, batch_id)

            # Create prediction records
            result = []
            for i, date in enumerate(future_dates):
                # Skip dates in the past
                if date < fields.Date.today():
                    continue

                # Get the prediction value and ensure it's non-negative
                try:
                    prediction_value = float(predictions[i])
                    prediction_value = max(1.0, prediction_value)  # Ensure positive value
                except (IndexError, ValueError) as e:
                    _logger.error(f"Error accessing prediction value: {str(e)}")
                    continue

                # Apply product-specific factors
                # 1. Price factor
                price_factor = 1.0
                if product_stats['avg_price'] > 0:
                    price_factor = 0.8 + (product_stats['avg_price'] / 1000)
                    price_factor = max(0.8, min(1.5, price_factor))

                # 2. Demand volume factor
                volume_factor = 1.0
                if product_stats['demand_count'] > 0:
                    volume_factor = 0.9 + (product_stats['demand_count'] / 100)
                    volume_factor = max(0.9, min(1.3, volume_factor))

                # 3. Customer diversity factor
                customer_factor = 1.0
                if product_stats['unique_customers'] > 0:
                    customer_factor = 0.9 + (product_stats['unique_customers'] / 20)
                    customer_factor = max(0.9, min(1.3, customer_factor))

                # 4. Lead time factor
                lead_time_factor = 1.0 + (lead_time / 30) * 0.1  # 10% extra for each month of lead time

                # Apply all factors
                prediction_value = prediction_value * price_factor * volume_factor * customer_factor * lead_time_factor

                # Add safety stock based on safety_stock_days
                safety_stock = 0
                if self.safety_stock_days > 0:
                    # Calculate daily average from prediction
                    daily_avg = prediction_value / 30  # Assuming monthly average
                    safety_stock = daily_avg * self.safety_stock_days
                    prediction_value += safety_stock

                # Adjust prediction value based on prediction period
                if self.prediction_period == 'weekly':
                    prediction_value *= 7  # Multiply by 7 for weekly
                elif self.prediction_period == 'monthly':
                    days_in_month = calendar.monthrange(date.year, date.month)[1]
                    prediction_value *= days_in_month  # Multiply by days in month
                elif self.prediction_period == 'quarterly':
                    quarter = (date.month - 1) // 3 + 1
                    days_in_quarter = 90 + (quarter == 1 or quarter == 4) * 2  # Approximate days in quarter
                    prediction_value *= days_in_quarter  # Multiply by days in quarter
                elif self.prediction_period == 'yearly':
                    days_in_year = 366 if calendar.isleap(date.year) else 365
                    prediction_value *= days_in_year  # Multiply by days in year

                # Calculate start and end dates for the period
                start_date, end_date = self._calculate_period_dates(date)

                result.append({
                    'date': date,
                    'product_id': product.id,
                    'warehouse_id': self.warehouse_id.id,
                    'prediction_value': prediction_value,
                    'prediction_period': self.prediction_period,
                    'model_id': self.model_id.id,
                    'generator_id': self.env.user.id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'name': self.forecast_name,
                    'description': f"{self.forecast_description} (ML model demand prediction)",
                    'batch_id': batch_id,
                    'includes_safety_stock': self.safety_stock_days > 0,
                    'safety_stock_days': self.safety_stock_days,
                })

            _logger.info(f"Generated {len(result)} ML model demand predictions for product {product.name}")
            return result

        except Exception as e:
            _logger.error(f"Error in _predict_demand_with_ml_model for {product.name}: {str(e)}")
            _logger.error(traceback.format_exc())
            return self._generate_zero_demand_prediction(product, batch_id)

    def _calculate_end_date(self):
        """Calculate the end date based on forecast days and period"""
        if self.prediction_period == 'daily':
            return self.forecast_start_date + timedelta(days=self.forecast_days - 1)
        elif self.prediction_period == 'weekly':
            weeks = max(1, self.forecast_days // 7)
            return self.forecast_start_date + timedelta(weeks=weeks)
        elif self.prediction_period == 'monthly':
            months = max(1, self.forecast_days // 30)
            return self.forecast_start_date + relativedelta(months=months)
        elif self.prediction_period == 'quarterly':
            quarters = max(1, self.forecast_days // 90)
            return self.forecast_start_date + relativedelta(months=3 * quarters)
        elif self.prediction_period == 'yearly':
            years = max(1, self.forecast_days // 365)
            return self.forecast_start_date + relativedelta(years=years)
        return self.forecast_start_date + timedelta(days=self.forecast_days - 1)

    def _get_historical_sales_data(self, products):
        """Get historical sales data for the selected products"""
        # Get sales data from the last 90 days
        date_from = fields.Date.today() - timedelta(days=90)

        # Get sales order lines
        domain = [
            ('product_id', 'in', products.ids),
            ('order_id.warehouse_id', '=', self.warehouse_id.id),
            ('order_id.date_order', '>=', date_from),
            ('order_id.state', 'in', ['sale', 'done']),
        ]

        sales_lines = self.env['sale.order.line'].search(domain)
        _logger.info(f"Found {len(sales_lines)} sales order lines for historical data")

        # Parepare data
        sales_data = []
        for line in sales_lines:
            sales_data.append({
                'date': line.order_id.date_order.date(),
                'product_id': line.product_id.id,
                'product_name': line.product_id.name,
                'quantity': line.product_uom_qty,
                'price': line.price_unit,
                'amount': line.price_subtotal,
                'categ_id': line.product_id.categ_id.id,
                'partner_id': line.order_id.partner_id.id,
                'order_id': line.order_id.id,
            })

        return pd.DataFrame(sales_data)

    def _get_historical_stockout_data(self, products):
        """Get historical stockout data for the selected products"""
        # Try to get stockout data from stock moves or inventory adjustments
        date_from = fields.Date.today() - timedelta(days=90)

        # Look for stock moves that might indicate stockouts
        # This is a simplified approach - in a real implementation, you would need
        # to define what constitutes a stockout in your specific Odoo setup
        domain = [
            ('product_id', 'in', products.ids),
            ('location_id.usage', '=', 'internal'),
            ('location_dest_id.usage', '=', 'customer'),
            ('state', '=', 'done'),
            ('date', '>=', date_from),
        ]

        stock_moves = self.env['stock.move'].search(domain)
        _logger.info(f"Found {len(stock_moves)} stock moves for stockout analysis")

        if not stock_moves:
            return None

        # Check for products that went out of stock
        stockout_data = []

        # Group stock moves by product and date
        for product in products:
            product_moves = stock_moves.filtered(lambda m: m.product_id.id == product.id)

            # Get inventory levels over time
            inventory_levels = {}

            # This is a simplified approach - in a real implementation, you would
            # need to calculate actual inventory levels over time
            for move in product_moves:
                move_date = move.date.date()
                if move_date not in inventory_levels:
                    inventory_levels[move_date] = product.with_context(to_date=move.date).qty_available

            # Look for dates where inventory went to zero or below
            for date, level in inventory_levels.items():
                if level <= 0:
                    # Estimate stockout quantity based on average daily sales
                    avg_daily_sales = sum(m.product_uom_qty for m in product_moves) / len(
                        product_moves) if product_moves else 0
                    stockout_quantity = avg_daily_sales * 0.2  # Assume 20% of average daily sales were lost

                    stockout_data.append({
                        'date': date,
                        'product_id': product.id,
                        'product_name': product.name,
                        'stockout_quantity': stockout_quantity,
                    })

        return pd.DataFrame(stockout_data) if stockout_data else None

    def _get_periods_to_forecast(self):
        """Calculate the number of periods to forecast based on prediction_period and forecast_days"""
        if self.prediction_period == 'daily':
            return self.forecast_days
        elif self.prediction_period == 'weekly':
            return max(1, self.forecast_days // 7)
        elif self.prediction_period == 'monthly':
            return max(1, self.forecast_days // 30)
        elif self.prediction_period == 'quarterly':
            return max(1, self.forecast_days // 90)
        elif self.prediction_period == 'yearly':
            return max(1, self.forecast_days // 365)
        return self.forecast_days  # Default to daily

    def _generate_future_dates(self, periods):
        """Generate future dates based on prediction period"""
        start_date = self.forecast_start_date
        dates = []

        for i in range(periods):
            if self.prediction_period == 'daily':
                date = start_date + timedelta(days=i)
            elif self.prediction_period == 'weekly':
                date = start_date + timedelta(weeks=i)
            elif self.prediction_period == 'monthly':
                date = start_date + relativedelta(months=i)
            elif self.prediction_period == 'quarterly':
                date = start_date + relativedelta(months=3 * i)
            elif self.prediction_period == 'yearly':
                date = start_date + relativedelta(years=i)
            else:  # Default to daily
                date = start_date + timedelta(days=i)

            dates.append(date)

        return dates

    def _calculate_period_dates(self, date):
        """Calculate start and end dates for a given period"""
        if self.prediction_period == 'daily':
            start_date = date
            end_date = date
        elif self.prediction_period == 'weekly':
            start_date = date
            end_date = date + timedelta(days=6)
        elif self.prediction_period == 'monthly':
            start_date = date
            # Last day of month
            next_month = date.replace(day=28) + timedelta(days=4)
            end_date = next_month - timedelta(days=next_month.day)
        elif self.prediction_period == 'quarterly':
            start_date = date
            end_date = date + relativedelta(months=3) - timedelta(days=1)
        elif self.prediction_period == 'yearly':
            start_date = date
            end_date = date + relativedelta(years=1) - timedelta(days=1)
        else:  # Default to daily
            start_date = date
            end_date = date

        return start_date, end_date

    def _create_demand_prediction_records(self, predictions):
        """Create demand prediction records from the predictions"""
        DemandPrediction = self.env['demand.prediction']
        created_count = 0

        for pred in predictions:
            try:
                # Create prediction record
                vals = {
                    'name': pred['name'],
                    'date': pred['date'],
                    'product_id': pred['product_id'],
                    'warehouse_id': pred['warehouse_id'],
                    'prediction_value': pred['prediction_value'],
                    'prediction_period': pred['prediction_period'],
                    'model_id': pred['model_id'],
                    'generator_id': pred['generator_id'],
                    'state': 'draft',
                    'start_date': pred['start_date'],
                    'end_date': pred['end_date'],
                    'description': pred['description'],
                    'batch_id': pred['batch_id'],
                    'includes_safety_stock': pred.get('includes_safety_stock', False),
                    'safety_stock_days': pred.get('safety_stock_days', 0),
                }

                DemandPrediction.create(vals)
                created_count += 1
            except Exception as e:
                _logger.error(f"Error creating demand prediction record: {str(e)}")
                _logger.error(f"Prediction data: {pred}")

        _logger.info(f"Created {created_count} demand prediction records out of {len(predictions)} predictions")


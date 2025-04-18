import logging
import base64
import pickle
import os
from odoo import api, fields, models, _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)


class SalesPredictionModel(models.Model):
    _name = 'sales.prediction.model'
    _description = 'Sales Prediction Model'

    name = fields.Char(string='Name', required=True)
    model_file = fields.Binary(string='Model File', attachment=True)
    model_filename = fields.Char(string='Model Filename')
    scaler_file = fields.Binary(string='Scaler File', attachment=True)
    scaler_filename = fields.Char(string='Scaler Filename')
    features_file = fields.Binary(string='Features File', attachment=True)
    features_filename = fields.Char(string='Features Filename')
    active = fields.Boolean(string='Active', default=True)
    create_date = fields.Datetime(string='Created On', readonly=True)
    create_uid = fields.Many2one('res.users', string='Created By', readonly=True)
    model_type = fields.Selection([
        ('xgboost', 'XGBoost'),
        ('prophet', 'Prophet'),
        ('sarima', 'SARIMA'),
        ('random_forest', 'Random Forest'),
        ('lightgbm', 'LightGBM'),
    ], string='Model Type', default='xgboost', required=True)
    description = fields.Text(string='Description')
    metrics = fields.Text(string='Performance Metrics')

    def _get_model_path(self):
        """Get the path to store the model temporarily"""
        module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(module_path, 'models')
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def load_model(self):
        """Load the model, scaler, and features from binary fields"""
        self.ensure_one()

        if not self.model_file:
            raise UserError(_("Model file is missing."))

        # For Prophet models, we only need the model file
        if self.model_type == 'prophet':
            model_dir = self._get_model_path()
            model_path = os.path.join(model_dir, 'temp_model.pkl')

            try:
                # Write binary data to file
                with open(model_path, 'wb') as f:
                    f.write(base64.b64decode(self.model_file))

                # Load the model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                return model, None, None

            except Exception as e:
                raise UserError(_(f"Error loading Prophet model: {str(e)}"))

            finally:
                # Clean up temporary file
                if os.path.exists(model_path):
                    os.remove(model_path)

        # For other model types, require all files
        if not self.scaler_file or not self.features_file:
            raise UserError(_("Scaler or features file is missing for this model type."))

        model_dir = self._get_model_path()

        # Save binary data to temporary files
        model_path = os.path.join(model_dir, 'temp_model.pkl')
        scaler_path = os.path.join(model_dir, 'temp_scaler.pkl')
        features_path = os.path.join(model_dir, 'temp_features.pkl')

        try:
            # Write binary data to files
            with open(model_path, 'wb') as f:
                f.write(base64.b64decode(self.model_file))

            with open(scaler_path, 'wb') as f:
                f.write(base64.b64decode(self.scaler_file))

            with open(features_path, 'wb') as f:
                f.write(base64.b64decode(self.features_file))

            # Load the model, scaler, and features
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            with open(features_path, 'rb') as f:
                features = pickle.load(f)

            return model, scaler, features

        except Exception as e:
            raise UserError(_(f"Error loading model: {str(e)}"))

        finally:
            # Clean up temporary files
            for path in [model_path, scaler_path, features_path]:
                if os.path.exists(path):
                    os.remove(path)


# -*- coding: utf-8 -*-

from odoo import models, fields, api, _
from odoo.exceptions import UserError
import logging
import time
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)


class RetrainDemandModelWizard(models.TransientModel):
    _name = 'retrain.demand.model.wizard'
    _description = 'Retrain Demand Model Wizard'

    model_id = fields.Many2one('demand.prediction.model', string='Model', required=True)
    date_from = fields.Date('From Date', required=True, default=lambda self: fields.Date.today() - timedelta(days=365))
    date_to = fields.Date('To Date', required=True, default=fields.Date.today)
    include_stockouts = fields.Boolean('Include Stockout Analysis', default=True)
    include_seasonality = fields.Boolean('Include Seasonality', default=True)
    include_lead_time = fields.Boolean('Include Lead Time', default=True)
    debug_mode = fields.Boolean('Debug Mode', default=False)

    @api.constrains('date_from', 'date_to')
    def _check_dates(self):
        for record in self:
            if record.date_from > record.date_to:
                raise UserError(_("From date must be before to date"))

    def action_retrain_model(self):
        """Retrain the selected model with new data"""
        self.ensure_one()

        start_time = time.time()

        try:
            # This is a placeholder for the actual retraining logic
            # In a real implementation, you would:
            # 1. Gather training data based on the date range
            # 2. Process the data (include stockouts, seasonality, lead time if selected)
            # 3. Train the model using the appropriate algorithm
            # 4. Save the model, scaler, and features files
            # 5. Update the model record

            # For now, we'll just log the operation and update the last_trained field
            self.model_id.write({
                'last_trained': fields.Datetime.now(),
            })

            # Log the operation
            execution_time = time.time() - start_time
            self.env['demand.prediction.log'].log_operation(
                'retrain',
                'success',
                f"Model {self.model_id.name} retrained successfully",
                execution_time,
                model_id=self.model_id.id
            )

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Success'),
                    'message': _('Model retrained successfully.'),
                    'sticky': False,
                    'type': 'success',
                }
            }

        except Exception as e:
            # Log the error
            execution_time = time.time() - start_time
            _logger.error(f"Error retraining model: {str(e)}")

            self.env['demand.prediction.log'].log_operation(
                'retrain',
                'failed',
                f"Error retraining model: {str(e)}",
                execution_time,
                model_id=self.model_id.id
            )

            raise UserError(_(f"Error retraining model: {str(e)}"))
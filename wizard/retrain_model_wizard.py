from odoo import models, fields, api
from odoo.exceptions import UserError
from odoo import _
import logging

_logger = logging.getLogger(__name__)


class RetrainModelWizard(models.TransientModel):
    _name = 'retrain.model.wizard'
    _description = 'Retrain Model Wizard'

    name = fields.Char(string='Model Name', required=True)
    model_id = fields.Many2one('sales.prediction.model', string='Base Model')
    model_type = fields.Selection([
        ('xgboost', 'XGBoost'),
        ('prophet', 'Prophet'),
        ('sarima', 'SARIMA'),
        ('random_forest', 'Random Forest'),
        ('lightgbm', 'LightGBM'),
    ], string='Model Type', default='xgboost', required=True)
    model_file = fields.Binary(string='Model File', required=True)
    model_filename = fields.Char(string='Model Filename')
    scaler_file = fields.Binary(string='Scaler File')
    scaler_filename = fields.Char(string='Scaler Filename')
    features_file = fields.Binary(string='Features File')
    features_filename = fields.Char(string='Features Filename')
    description = fields.Text(string='Description')

    @api.onchange('model_type')
    def _onchange_model_type(self):
        """Update required fields based on model type"""
        if self.model_type == 'prophet':
            # Clear scaler and features files as they're not needed for Prophet
            self.scaler_file = False
            self.features_file = False
            return {'warning': {
                'title': _('Prophet Model Selected'),
                'message': _('Prophet models only require the model file, not scaler or features files.')
            }}

    def action_import_model(self):
        """Import a pre-trained model"""
        self.ensure_one()

        # Validate required files based on model type
        if self.model_type == 'prophet':
            if not self.model_file:
                raise UserError(_("Please upload the Prophet model file."))
        else:
            # For other model types, require all files
            if not self.model_file or not self.scaler_file or not self.features_file:
                raise UserError(_("Please upload all required files: model, scaler, and features."))

        try:
            # Create a new model record
            model_vals = {
                'name': self.name,
                'model_file': self.model_file,
                'model_filename': self.model_filename,
                'model_type': self.model_type,
                'description': self.description or f"Imported {self.model_type} model",
            }

            # Add scaler and features files only for non-Prophet models
            if self.model_type != 'prophet':
                model_vals.update({
                    'scaler_file': self.scaler_file,
                    'scaler_filename': self.scaler_filename,
                    'features_file': self.features_file,
                    'features_filename': self.features_filename,
                })

            model = self.env['sales.prediction.model'].create(model_vals)

            # Log the operation
            self.env['sales.prediction.log'].log_operation(
                'import',
                'success',
                f"Imported {self.model_type} model: {self.name}",
                0.0
            )

            return {
                'type': 'ir.actions.act_window',
                'name': _('Sales Prediction Model'),
                'res_model': 'sales.prediction.model',
                'res_id': model.id,
                'view_mode': 'form',
                'target': 'current',
            }

        except Exception as e:
            # Log the error
            self.env['sales.prediction.log'].log_operation(
                'import',
                'failed',
                f"Error importing model: {str(e)}",
                0.0
            )
            _logger.error(f"Error importing model: {str(e)}")
            raise UserError(_(f"Error importing model: {str(e)}"))
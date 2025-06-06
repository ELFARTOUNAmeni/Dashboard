# -*- coding: utf-8 -*-

from odoo import models, fields, api, _
from odoo.exceptions import AccessError, UserError, ValidationError
import logging

_logger = logging.getLogger(__name__)


class SalesPredictionBatch(models.Model):
    _name = 'sales.prediction.batch'
    _description = 'Sales Prediction Batch'
    _order = 'create_date desc'
    _inherit = ['mail.thread', 'mail.activity.mixin']  # Add mail thread support for tracking

    name = fields.Char(string='Batch Name', required=True, tracking=True)
    description = fields.Text('Description', tracking=True)
    date = fields.Date('Generation Date', default=fields.Date.today, required=True, tracking=True)
    prediction_count = fields.Integer('Prediction Count', compute='_compute_prediction_count')
    user_id = fields.Many2one('res.users', string='Generated By', default=lambda self: self.env.user, required=True, tracking=True)
    prediction_period = fields.Selection([
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('quarterly', 'Quarterly'),
        ('yearly', 'Yearly'),
    ], string='Prediction Period', required=True, tracking=True)
    model_id = fields.Many2one('sales.prediction.model', string='Model Used', required=True, tracking=True)
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse', required=True, tracking=True)
    start_date = fields.Date('Start Date', required=True, tracking=True)
    end_date = fields.Date('End Date', tracking=True)
    state = fields.Selection([
        ('draft', 'Draft'),
        ('confirmed', 'Confirmed'),
        ('cancelled', 'Cancelled')
    ], string='Status', default='draft', required=True, tracking=True)

    def _compute_prediction_count(self):
        for batch in self:
            batch.prediction_count = self.env['sales.prediction'].search_count([
                ('batch_id', '=', batch.id)
            ])

    def action_view_predictions(self):
        self.ensure_one()
        return {
            'name': _('Predictions'),
            'type': 'ir.actions.act_window',
            'res_model': 'sales.prediction',
            'view_mode': 'tree,form',
            'domain': [('batch_id', '=', self.id)],
            'context': {'default_batch_id': self.id},
        }

    def action_confirm(self):
        self.ensure_one()
        # Check if user is in admin group
        if not self.env.user.has_group('sales_prediction.group_sales_prediction_admin'):
            raise AccessError(_("Only administrators can change prediction batch states."))
        self.write({'state': 'confirmed'})
        self.env['sales.prediction'].search([('batch_id', '=', self.id)]).write({'state': 'confirmed'})

    def action_cancel(self):
        self.ensure_one()
        # Check if user is in admin group
        if not self.env.user.has_group('sales_prediction.group_sales_prediction_admin'):
            raise AccessError(_("Only administrators can change prediction batch states."))
        self.write({'state': 'cancelled'})
        self.env['sales.prediction'].search([('batch_id', '=', self.id)]).write({'state': 'cancelled'})

    def action_reset_to_draft(self):
        self.ensure_one()
        # Check if user is in admin group
        if not self.env.user.has_group('sales_prediction.group_sales_prediction_admin'):
            raise AccessError(_("Only administrators can change prediction batch states."))
        self.write({'state': 'draft'})
        self.env['sales.prediction'].search([('batch_id', '=', self.id)]).write({'state': 'draft'})


class SalesPrediction(models.Model):
    _name = 'sales.prediction'
    _description = 'Sales Prediction'
    _inherit = ['mail.thread', 'mail.activity.mixin']  # Add mail thread support for tracking

    name = fields.Char(string='Name', default='New', required=True, tracking=True)
    date = fields.Date('Date', required=True, tracking=True)
    product_id = fields.Many2one('product.product', string='Product', required=True, tracking=True)
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse', required=True, tracking=True)
    prediction_value = fields.Float('Prediction Value', required=True, tracking=True)
    prediction_period = fields.Selection([
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('quarterly', 'Quarterly'),
        ('yearly', 'Yearly'),
    ], string='Prediction Period', required=True, tracking=True)
    model_id = fields.Many2one('sales.prediction.model', string='Model', required=True, tracking=True)
    # Add batch_id field to link predictions to batches
    batch_id = fields.Many2one('sales.prediction.batch', string='Prediction Batch', ondelete='cascade', tracking=True)
    # Add state field
    state = fields.Selection([
        ('draft', 'Draft'),
        ('confirmed', 'Confirmed'),
        ('cancelled', 'Cancelled')
    ], string='Status', default='draft', required=True, tracking=True)
    # Add generator_id field
    generator_id = fields.Many2one('res.users', string='Generated By', tracking=True)
    # Add date range fields for non-daily predictions
    start_date = fields.Date('Start Date', tracking=True)
    end_date = fields.Date('End Date', tracking=True)
    # Add description field
    description = fields.Text('Description', help="Additional details about this forecast", tracking=True)


class SalesPredictionLog(models.Model):
    _name = 'sales.prediction.log'
    _description = 'Sales Prediction Log'
    _order = 'create_date desc'

    operation = fields.Selection([
        ('train', 'Model Training'),
        ('predict', 'Prediction Generation'),
        ('update_actual', 'Update Actual Values'),
        ('generate', 'Batch Generation'),
        ('import', 'Import Model'),
    ], string='Operation', required=True)
    status = fields.Selection([
        ('success', 'Success'),
        ('failed', 'Failed'),
    ], string='Status', required=True)
    message = fields.Text('Message')
    execution_time = fields.Float('Execution Time (s)')
    user_id = fields.Many2one('res.users', string='User', default=lambda self: self.env.user)
    batch_id = fields.Many2one('sales.prediction.batch', string='Prediction Batch')
    model_id = fields.Many2one('sales.prediction.model', string='Model')
    create_date = fields.Datetime('Creation Date', readonly=True)
    product_count = fields.Integer('Product Count')
    prediction_count = fields.Integer('Prediction Count')

    @api.model
    def log_operation(self, operation, status, message, execution_time, batch_id=False, model_id=False,
                      product_count=0, prediction_count=0, metrics=None):
        """
        Log an operation

        Parameters:
        -----------
        operation : str
            Type of operation performed
        status : str
            Status of the operation (success/failed)
        message : str
            Detailed message about the operation
        execution_time : float
            Time taken to execute the operation in seconds
        batch_id : int, optional
            ID of the related prediction batch
        model_id : int, optional
            ID of the related prediction model
        product_count : int, optional
            Number of products processed
        prediction_count : int, optional
            Number of predictions generated
        metrics : dict, optional
            Dictionary containing performance metrics (not used in simplified version)
        """
        vals = {
            'operation': operation,
            'status': status,
            'message': message,
            'execution_time': execution_time,
            'batch_id': batch_id,
            'model_id': model_id,
            'product_count': product_count,
            'prediction_count': prediction_count,
        }

        return self.create(vals)
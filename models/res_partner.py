from odoo import models, fields, api
from datetime import datetime, timedelta
import logging

_logger = logging.getLogger(__name__)


class ResPartner(models.Model):
    _inherit = 'res.partner'

    # Fields
    partner_gid = fields.Integer(string='Company database ID', readonly=True, copy=False)
    additional_info = fields.Text(string='Additional Information')
    segment_id = fields.Many2one('customer.segment', string='Customer Segment', readonly=True)

    # Computed Metrics
    total_spent = fields.Float(compute='_compute_customer_metrics', store=True)
    order_count = fields.Integer(compute='_compute_customer_metrics', store=True)
    avg_order_value = fields.Float(compute='_compute_customer_metrics', store=True)
    days_since_first_order = fields.Integer(compute='_compute_customer_metrics', store=True)
    days_since_last_order = fields.Integer(compute='_compute_customer_metrics', store=True)
    order_frequency = fields.Float(compute='_compute_customer_metrics', store=True)
    category_count = fields.Integer(compute='_compute_customer_metrics', store=True)

    @api.depends('sale_order_ids', 'sale_order_ids.state',
                 'sale_order_ids.amount_total', 'sale_order_ids.date_order')
    def _compute_customer_metrics(self):
        for partner in self:
            valid_orders = partner.sale_order_ids.filtered(
                lambda o: o.state in ['sale', 'done'] and o.date_order
            ).sorted('date_order')

            partner.order_count = len(valid_orders)
            partner.total_spent = sum(valid_orders.mapped('amount_total'))

            if partner.order_count > 0:
                partner.avg_order_value = partner.total_spent / partner.order_count

                first_date = valid_orders[0].date_order
                last_date = valid_orders[-1].date_order
                today = fields.Datetime.now()

                partner.days_since_first_order = (today - first_date).days
                partner.days_since_last_order = (today - last_date).days

                if partner.order_count > 1:
                    total_days = (last_date - first_date).days
                    partner.order_frequency = total_days / (partner.order_count - 1)
                else:
                    partner.order_frequency = 0.0

                categories = valid_orders.order_line.product_id.categ_id
                partner.category_count = len(categories.ids)
            else:
                partner.update({
                    'total_spent': 0.0,
                    'avg_order_value': 0.0,
                    'days_since_first_order': 0,
                    'days_since_last_order': 0,
                    'order_frequency': 0.0,
                    'category_count': 0
                })

    def action_recompute_metrics(self):
        self._compute_customer_metrics()
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Success',
                'message': f'Recomputed metrics for {len(self)} customers',
                'type': 'success',
                'sticky': False,
            }
        }

    def get_recommendations(self):
        self.ensure_one()
        if not self.segment_id:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'No Segment',
                    'message': 'This customer has not been assigned to a segment yet.',
                    'sticky': False,
                    'type': 'warning',
                }
            }
        action = self.env.ref('sales_prediction.action_get_recommendations_wizard').read()[0]
        action['context'] = {'default_partner_id': self.id}
        return action

    def clean_future_orders_and_recompute(self):
        """Corrige les commandes futures en les remplaçant par une date d'il y a 1 mois, et recalcule les métriques client."""
        today = fields.Datetime.now()
        one_month_ago = today - timedelta(days=30)

        # CORRECTION: Utiliser '>' au lieu de '=' pour trouver toutes les commandes futures
        future_orders = self.env['sale.order'].search([
            ('date_order', '>', today),  # <-- Ceci est la correction clé
            ('state', 'in', ['sale', 'done'])
        ])

        _logger.info(f"Found {len(future_orders)} orders with future dates.")

        for order in future_orders:
            _logger.info(f"Correcting order {order.name} (Partner: {order.partner_id.name}) from {order.date_order} to {one_month_ago}")
            order.sudo().write({'date_order': one_month_ago})

        affected_partners = future_orders.mapped('partner_id')
        if affected_partners:
            affected_partners._compute_customer_metrics()
            _logger.info(f"Recomputed metrics for {len(affected_partners)} partners")

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Future Orders Corrected',
                'message': f'Corrected {len(future_orders)} orders with future dates to one month ago.',
                'type': 'success',
                'sticky': False,
            }
        }
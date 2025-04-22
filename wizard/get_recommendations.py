from odoo import models, fields, api


class GetRecommendationsWizard(models.TransientModel):
    _name = 'get.recommendations.wizard'
    _description = 'Get Product Recommendations'

    partner_id = fields.Many2one('res.partner', string='Customer', required=True)
    segment_id = fields.Many2one('customer.segment', related='partner_id.segment_id', string='Segment')
    recommendation_ids = fields.Many2many('product.recommendation', string='Recommendations',
                                          compute='_compute_recommendations')
    recommendation_count = fields.Integer(string='Number of Recommendations', compute='_compute_recommendations')

    @api.depends('partner_id', 'segment_id')
    def _compute_recommendations(self):
        for wizard in self:
            if wizard.segment_id:
                wizard.recommendation_ids = wizard.segment_id.recommendation_ids
                wizard.recommendation_count = len(wizard.recommendation_ids)
            else:
                wizard.recommendation_ids = False
                wizard.recommendation_count = 0

    def action_add_to_cart(self):
        """Add recommended products to a new quotation"""
        self.ensure_one()

        if not self.recommendation_ids:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'No Recommendations',
                    'message': 'No product recommendations available for this customer',
                    'sticky': False,
                    'type': 'warning',
                }
            }

        # Create a new quotation
        order = self.env['sale.order'].create({
            'partner_id': self.partner_id.id,
            'note': 'Created from product recommendations',
        })

        # Add recommended products
        for rec in self.recommendation_ids:
            self.env['sale.order.line'].create({
                'order_id': order.id,
                'product_id': rec.product_id.id,
                'product_uom_qty': 1,
            })

        # Open the quotation
        return {
            'type': 'ir.actions.act_window',
            'res_model': 'sale.order',
            'res_id': order.id,
            'view_mode': 'form',
            'target': 'current',
        }

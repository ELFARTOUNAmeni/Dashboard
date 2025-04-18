from odoo import models, fields, api


class ProductRecommendation(models.Model):
    _name = 'product.recommendation'
    _description = 'Product Recommendation'
    _order = 'score desc'

    segment_id = fields.Many2one('customer.segment', string='Customer Segment', required=True, ondelete='cascade')
    product_id = fields.Many2one('product.product', string='Product', required=True)
    score = fields.Float(string='Popularity Score', help='Higher score means more popular in this segment')
    purchase_count = fields.Integer(string='Purchase Count',
                                    help='Number of times purchased by customers in this segment')

    _sql_constraints = [
        ('segment_product_uniq', 'unique(segment_id, product_id)', 'Product recommendation must be unique per segment!')
    ]

from odoo import models, fields, api


class CustomerSegment(models.Model):
    _name = 'customer.segment'
    _description = 'Customer Segment'

    name = fields.Char(string='Segment Name', required=True)
    description = fields.Text(string='Description')
    customer_count = fields.Integer(string='Number of Customers', compute='_compute_customer_count')
    partner_ids = fields.One2many('res.partner', 'segment_id', string='Customers')
    recommendation_ids = fields.One2many('product.recommendation', 'segment_id', string='Recommended Products')
    active = fields.Boolean(default=True)

    # Segment characteristics (centroids from K-means)
    avg_order_value = fields.Float(string='Average Order Value', digits=(16, 2))
    avg_order_frequency = fields.Float(string='Average Order Frequency (days)', digits=(16, 2))
    avg_product_categories = fields.Integer(string='Average Product Categories')
    last_generated = fields.Datetime(string='Last Generated')

    # Ajout du champ algorithm pour GMM
    algorithm = fields.Selection([
        ('kmeans', 'K-Means'),
        ('gmm', 'Gaussian Mixture Model')
    ], string='Algorithm Used', readonly=True, default='kmeans')

    @api.depends('partner_ids')
    def _compute_customer_count(self):
        for segment in self:
            segment.customer_count = len(segment.partner_ids)

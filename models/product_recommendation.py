from odoo import models, fields, api

class ProductRecommendation(models.Model):
    _name = 'product.recommendation'
    _description = 'Product Recommendation'
    _order = 'score desc'

    segment_id = fields.Many2one('customer.segment', string='Customer Segment', required=True, ondelete='cascade')
    product_id = fields.Many2one('product.product', string='Product', required=True)
    score = fields.Float(string='Recommendation Score', help='Higher score indicates stronger recommendation')
    purchase_count = fields.Integer(string='Purchase Count',
                                   help='Number of times purchased by customers in this segment')
    algorithm = fields.Selection([
        ('svd', 'Collaborative Filtering (SVD)'),
        ('nmf', 'Non-negative Matrix Factorization'),
        ('content', 'Content-Based Filtering'),
        ('hybrid', 'Hybrid Approach'),
        ('knn', 'K-Nearest Neighbors'),
        ('ml_popularity', 'Enhanced Popularity Model')
    ], string='ML Algorithm Used', default='svd',
       help='Machine learning algorithm used to generate this recommendation')

    _sql_constraints = [
        ('segment_product_uniq', 'unique(segment_id, product_id)', 'Product recommendation must be unique per segment!')
    ]
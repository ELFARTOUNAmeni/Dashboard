from odoo import models, fields, api, tools
import logging

_logger = logging.getLogger(__name__)


class CustomerSegmentDashboard(models.Model):
    _name = 'customer.segment.dashboard'
    _description = 'Customer Segment Dashboard'
    _auto = False  # This is a SQL view, not a real table

    name = fields.Char(string='Dashboard Name', default='Customer Segmentation Dashboard')

    def init(self):
        # This is a SQL view initialization
        # We're not creating a real table, just a placeholder for the dashboard
        query = """
            SELECT 1 as id, 'Customer Segmentation Dashboard' as name
        """
        tools.drop_view_if_exists(self.env.cr, self._table)
        self.env.cr.execute(f"""CREATE or REPLACE VIEW {self._table} as ({query})""")

    @api.model
    def get_segment_summary(self):
        """Get summary data for all segments"""
        segments = self.env['customer.segment'].search([('active', '=', True)])

        return {
            'total_segments': len(segments),
            'total_customers': sum(segments.mapped('customer_count')),
            'segments': [{
                'id': segment.id,
                'name': segment.name,
                'customer_count': segment.customer_count,
                'avg_order_value': segment.avg_order_value,
                'avg_order_frequency': segment.avg_order_frequency,
            } for segment in segments]
        }

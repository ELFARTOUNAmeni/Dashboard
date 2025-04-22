from odoo import http
from odoo.http import request
import json
from datetime import datetime, timedelta


class CustomerSegmentationDashboard(http.Controller):
    @http.route('/sales_prediction/dashboard_data', type='json', auth='user')
    def get_dashboard_data(self, **kw):
        """Provide data for the customer segmentation dashboard"""
        user = request.env.user
        if not user.has_group('sales_prediction.group_sales_prediction_user'):
            return {'error': 'Access Denied'}

        # Get segments data
        segments = request.env['customer.segment'].search([('active', '=', True)])

        # Basic segment data
        segments_data = [{
            'id': segment.id,
            'name': segment.name,
            'customer_count': segment.customer_count,
            'avg_order_value': segment.avg_order_value,
            'avg_order_frequency': segment.avg_order_frequency,
            'avg_product_categories': segment.avg_product_categories,
            'algorithm': segment.algorithm,
            'last_generated': segment.last_generated.strftime('%Y-%m-%d %H:%M:%S') if segment.last_generated else '',
            'color': f"#{hash(segment.name) % 0xffffff:06x}"  # Generate consistent color based on name
        } for segment in segments]

        # Get total customers
        total_customers = request.env['res.partner'].search_count([
            ('customer_rank', '>', 0),
            ('active', '=', True)
        ])

        # Get segmented customers
        segmented_customers = request.env['res.partner'].search_count([
            ('customer_rank', '>', 0),
            ('segment_id', '!=', False),
            ('active', '=', True)
        ])

        # Get top products by segment
        top_products_by_segment = {}
        for segment in segments:
            recommendations = request.env['product.recommendation'].search([
                ('segment_id', '=', segment.id)
            ], limit=5, order='score desc')

            top_products_by_segment[segment.id] = [{
                'product_name': rec.product_id.name,
                'score': rec.score,
                'product_id': rec.product_id.id
            } for rec in recommendations]

        # Get recent orders by segment (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_orders_by_segment = {}

        for segment in segments:
            partners = request.env['res.partner'].search([('segment_id', '=', segment.id)])
            partner_ids = partners.ids

            if partner_ids:
                orders = request.env['sale.order'].search([
                    ('partner_id', 'in', partner_ids),
                    ('date_order', '>=', thirty_days_ago),
                    ('state', 'in', ['sale', 'done'])
                ])

                recent_orders_by_segment[segment.id] = {
                    'count': len(orders),
                    'total': sum(orders.mapped('amount_total')),
                    'avg': sum(orders.mapped('amount_total')) / len(orders) if orders else 0
                }
            else:
                recent_orders_by_segment[segment.id] = {
                    'count': 0,
                    'total': 0,
                    'avg': 0
                }

        # Get customer metrics distribution
        metrics = ['avg_order_value', 'order_frequency', 'total_spent', 'category_count']
        metrics_distribution = {}

        for metric in metrics:
            metrics_distribution[metric] = {}
            for segment in segments:
                partners = request.env['res.partner'].search([('segment_id', '=', segment.id)])
                if partners:
                    values = partners.mapped(metric)
                    metrics_distribution[metric][segment.id] = {
                        'avg': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0,
                    }
                else:
                    metrics_distribution[metric][segment.id] = {
                        'avg': 0,
                        'min': 0,
                        'max': 0,
                    }

        # Monthly sales trend by segment (last 6 months)
        months = []
        for i in range(5, -1, -1):
            date = datetime.now() - timedelta(days=30 * i)
            months.append(date.strftime('%Y-%m'))

        monthly_sales = {}
        for segment in segments:
            monthly_sales[segment.id] = {month: 0 for month in months}

            partners = request.env['res.partner'].search([('segment_id', '=', segment.id)])
            partner_ids = partners.ids

            if partner_ids:
                six_months_ago = datetime.now() - timedelta(days=180)
                orders = request.env['sale.order'].search([
                    ('partner_id', 'in', partner_ids),
                    ('date_order', '>=', six_months_ago),
                    ('state', 'in', ['sale', 'done'])
                ])

                for order in orders:
                    month_key = order.date_order.strftime('%Y-%m')
                    if month_key in monthly_sales[segment.id]:
                        monthly_sales[segment.id][month_key] += order.amount_total

        return {
            'segments': segments_data,
            'total_customers': total_customers,
            'segmented_customers': segmented_customers,
            'top_products_by_segment': top_products_by_segment,
            'recent_orders_by_segment': recent_orders_by_segment,
            'metrics_distribution': metrics_distribution,
            'monthly_sales': monthly_sales,
            'months': months
        }

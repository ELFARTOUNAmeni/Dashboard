from odoo import http
from odoo.http import request
from datetime import datetime, timedelta
import logging

_logger = logging.getLogger(__name__)

class SalesDashboardController(http.Controller):

    @http.route('/sales_prediction/sales_dashboard_data', type='json', auth='user')
    def get_sales_dashboard_data(self, **kw):
        user = request.env.user
        if not user.has_group('sales_prediction.group_sales_prediction_user'):
            return {'error': 'Access Denied'}

        date_from = kw.get('date_from', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        date_to = kw.get('date_to', datetime.now().strftime('%Y-%m-%d'))

        warehouse_id = kw.get('warehouse_id', False)
        category_id = kw.get('category_id', False)

        warehouse_domain = [('warehouse_id', '=', warehouse_id)] if warehouse_id else []
        category_domain = [('category_id', '=', category_id)] if category_id else []
        base_domain = [('date', '>=', date_from), ('date', '<=', date_to)]
        domain = base_domain + warehouse_domain + category_domain

        sales_data = request.env['sales.dashboard'].search(domain)

        top_products = request.env['sales.dashboard'].search(
            domain, limit=10, order='qty_sold desc'
        )

        top_products_data = [{
            'id': product.id,
            'name': product.name_get()[0][1],
            'qty_sold': product.qty_sold,
            'amount_sold': product.amount_sold,
            'current_stock': product.current_stock,
            'prediction_value': product.prediction_value,
            'color': f"#{hash(product.name_get()[0][1]) % 0xffffff:06x}"
        } for product in top_products]

        category_data = request.env['sales.dashboard'].read_group(
            domain,
            ['category_id', 'qty_sold', 'amount_sold'],
            ['category_id']
        )

        categories = [{
            'id': item['category_id'][0] if item.get('category_id') else 0,
            'name': item['category_id'][1] if item.get('category_id') else 'Uncategorized',
            'qty_sold': item['qty_sold'],
            'amount_sold': item['amount_sold'],
            'color': f"#{hash(item['category_id'][1] if item.get('category_id') else 'Uncategorized') % 0xffffff:06x}"
        } for item in category_data]

        warehouse_data = request.env['sales.dashboard'].read_group(
            domain,
            ['warehouse_id', 'qty_sold', 'amount_sold'],
            ['warehouse_id']
        )

        warehouses = [{
            'id': item['warehouse_id'][0] if item.get('warehouse_id') else 0,
            'name': item['warehouse_id'][1] if item.get('warehouse_id') else 'No Warehouse',
            'qty_sold': item['qty_sold'],
            'amount_sold': item['amount_sold'],
            'color': f"#{hash(item['warehouse_id'][1] if item.get('warehouse_id') else 'No Warehouse') % 0xffffff:06x}"
        } for item in warehouse_data]

        trend_data = request.env['sales.dashboard'].read_group(
            domain,
            ['date', 'qty_sold', 'amount_sold'],
            ['date:day']
        )

        dates = []
        quantities = []
        amounts = []

        for item in trend_data:
            date_str = item.get('date:day')
            dates.append(date_str)
            quantities.append(item['qty_sold'])
            amounts.append(item['amount_sold'])

        stock_status_data = request.env['sales.stock.comparison.dashboard'].read_group(
            category_domain + warehouse_domain,
            ['stock_status', 'sales_quantity', 'stock_quantity', 'sales_amount'],
            ['stock_status']
        )

        stock_status = [{
            'status': item['stock_status'] or 'unknown',
            'status_name': dict(request.env['sales.stock.comparison.dashboard']._fields['stock_status'].selection).get(
                item['stock_status'], 'Unknown'),
            'sales_quantity': item['sales_quantity'],
            'stock_quantity': item['stock_quantity'],
            'sales_amount': item['sales_amount'],
        } for item in stock_status_data]

        seasonal_data = request.env['seasonal.sales.analysis'].read_group(
            category_domain + warehouse_domain,
            ['season', 'avg_quantity', 'avg_amount', 'total_sales'],
            ['season']
        )

        seasons = [{
            'season': item['season'] or 'unknown',
            'season_name': dict(request.env['seasonal.sales.analysis']._fields['season'].selection).get(item['season'], 'Unknown'),
            'avg_quantity': item['avg_quantity'],
            'avg_amount': item['avg_amount'],
            'total_sales': item['total_sales'],
            'color': f"#{hash(item['season'] or 'unknown') % 0xffffff:06x}"
        } for item in seasonal_data]

        day_data = request.env['day.of.week.sales.analysis'].read_group(
            category_domain + warehouse_domain,
            ['day_of_week', 'avg_quantity', 'avg_amount', 'total_sales'],
            ['day_of_week']
        )

        days_of_week = [{
            'day': str(item['day_of_week']),
            'name': dict(request.env['day.of.week.sales.analysis']._fields['day_of_week'].selection).get(item['day_of_week'], 'Unknown'),
            'avg_quantity': item['avg_quantity'],
            'avg_amount': item['avg_amount'],
            'total_sales': item['total_sales'],
            'color': f"#{hash(str(item['day_of_week'])) % 0xffffff:06x}"
        } for item in day_data]

        days_of_week.sort(key=lambda x: int(x['day']))

        total_sales = sum(item.amount_sold for item in sales_data)
        total_quantity = sum(item.qty_sold for item in sales_data)
        avg_order_value = total_sales / len(sales_data) if sales_data else 0

        low_stock_products = request.env['sales.stock.comparison.dashboard'].search(
            [('stock_status', '=', 'low')] + category_domain + warehouse_domain,
            limit=10, order='sales_quantity desc'
        )

        low_stock_data = [{
            'id': product.id,
            'name': product.name_get()[0][1],
            'sales_quantity': product.sales_quantity,
            'stock_quantity': product.stock_quantity,
            'category_id': product.category_id.id if product.category_id else False,
            'category_name': product.category_id.name if product.category_id else 'Uncategorized',
        } for product in low_stock_products]

        all_warehouses = request.env['stock.warehouse'].search([])
        warehouse_filters = [{'id': w.id, 'name': w.name} for w in all_warehouses]

        all_categories = request.env['product.category'].search([])
        category_filters = [{'id': c.id, 'name': c.name} for c in all_categories]

        return {
            'top_products': top_products_data,
            'categories': categories,
            'warehouses': warehouses,
            'trend_dates': dates,
            'trend_quantities': quantities,
            'trend_amounts': amounts,
            'stock_status': stock_status,
            'seasons': seasons,
            'days_of_week': days_of_week,
            'total_sales': total_sales,
            'total_quantity': total_quantity,
            'avg_order_value': avg_order_value,
            'low_stock_products': low_stock_data,
            'warehouse_filters': warehouse_filters,
            'category_filters': category_filters
        }

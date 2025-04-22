# -*- coding: utf-8 -*-
from odoo import models, fields, api, _
from datetime import datetime, timedelta
import logging
import psycopg2

_logger = logging.getLogger(__name__)

from odoo import models, fields, api, tools

class SalesDashboard(models.Model):
    _name = 'sales.dashboard'
    _description = 'Sales Dashboard'
    _auto = False
    _order = 'qty_sold desc'

    name = fields.Char(string='Product Name')
    product_id = fields.Many2one('product.product', string='Product')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    category_id = fields.Many2one('product.category', string='Product Category')
    qty_sold = fields.Float(string='Quantity Sold')
    amount_sold = fields.Float(string='Amount Sold')
    date = fields.Date(string='Date')
    current_stock = fields.Float(string='Current Stock')
    prediction_value = fields.Float(string='Predicted Sales')

    def init(self):
        """Initialize the view"""
        # Handle both table and view cases
        try:
            self._cr.execute("DROP VIEW IF EXISTS sales_dashboard CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS sales_dashboard CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view
        self._cr.execute("""
            CREATE OR REPLACE VIEW sales_dashboard AS (
                SELECT
                    ROW_NUMBER() OVER() as id,
                    p.id as product_id,
                    w.id as warehouse_id,
                    pt.categ_id as category_id,
                    pt.name as name,
                    COALESCE(SUM(sol.product_uom_qty), 0) as qty_sold,
                    COALESCE(SUM(sol.price_subtotal), 0) as amount_sold,
                    so.date_order::date as date,
                    COALESCE(sq.quantity, 0) as current_stock,
                    COALESCE(sp.prediction_value, 0) as prediction_value
                FROM
                    product_product p
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                LEFT JOIN
                    sale_order_line sol ON sol.product_id = p.id
                LEFT JOIN
                    sale_order so ON so.id = sol.order_id
                LEFT JOIN
                    stock_warehouse w ON w.id = so.warehouse_id
                LEFT JOIN
                    (
                        SELECT 
                            product_id, 
                            SUM(quantity) as quantity 
                        FROM 
                            stock_quant 
                        WHERE 
                            location_id IN (SELECT id FROM stock_location WHERE usage = 'internal')
                        GROUP BY 
                            product_id
                    ) sq ON sq.product_id = p.id
                LEFT JOIN
                    sales_prediction sp ON sp.product_id = p.id AND sp.date = CURRENT_DATE
                WHERE
                    so.state in ('sale', 'done')
                GROUP BY
                    p.id, pt.id, pt.categ_id, w.id, so.date_order::date, sq.quantity, sp.prediction_value
            )
        """)

class ProductStockDashboard(models.Model):
    _name = 'product.stock.dashboard'
    _description = 'Product Stock Dashboard'
    _auto = False
    _order = 'quantity desc'

    product_id = fields.Many2one('product.product', string='Product')
    name = fields.Char(string='Product Name')
    category_id = fields.Many2one('product.category', string='Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    location_id = fields.Many2one('stock.location', string='Location')
    quantity = fields.Float(string='Quantity On Hand')
    reserved_quantity = fields.Float(string='Reserved Quantity')
    available_quantity = fields.Float(string='Available Quantity')

    def init(self):
        """Initialize the view"""
        # Handle both table and view cases
        try:
            self._cr.execute("DROP VIEW IF EXISTS product_stock_dashboard CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS product_stock_dashboard CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view
        self._cr.execute("""
            CREATE OR REPLACE VIEW product_stock_dashboard AS (
                SELECT
                    ROW_NUMBER() OVER() as id,
                    p.id as product_id,
                    pt.name as name,
                    pt.categ_id as category_id,
                    sl.warehouse_id as warehouse_id,
                    sq.location_id as location_id,
                    SUM(sq.quantity) as quantity,
                    SUM(sq.reserved_quantity) as reserved_quantity,
                    SUM(sq.quantity - sq.reserved_quantity) as available_quantity
                FROM
                    stock_quant sq
                JOIN
                    product_product p ON p.id = sq.product_id
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                JOIN
                    stock_location sl ON sl.id = sq.location_id
                WHERE
                    sl.usage = 'internal'
                GROUP BY
                    p.id, pt.id, pt.categ_id, sl.warehouse_id, sq.location_id
            )
        """)

class SalesHistoryDashboard(models.Model):
    _name = 'sales.history.dashboard'
    _description = 'Sales History Dashboard'
    _auto = False
    _order = 'date desc'

    date = fields.Date(string='Date')
    product_id = fields.Many2one('product.product', string='Product')
    category_id = fields.Many2one('product.category', string='Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    partner_id = fields.Many2one('res.partner', string='Customer')
    quantity = fields.Float(string='Quantity')
    amount = fields.Float(string='Amount')

    def init(self):
        """Initialize the view"""
        # Handle both table and view cases
        try:
            self._cr.execute("DROP VIEW IF EXISTS sales_history_dashboard CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS sales_history_dashboard CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view
        self._cr.execute("""
            CREATE OR REPLACE VIEW sales_history_dashboard AS (
                SELECT
                    ROW_NUMBER() OVER() as id,
                    so.date_order::date as date,
                    sol.product_id as product_id,
                    pt.categ_id as category_id,
                    so.warehouse_id as warehouse_id,
                    so.partner_id as partner_id,
                    SUM(sol.product_uom_qty) as quantity,
                    SUM(sol.price_subtotal) as amount
                FROM
                    sale_order_line sol
                JOIN
                    sale_order so ON so.id = sol.order_id
                JOIN
                    product_product p ON p.id = sol.product_id
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                WHERE
                    so.state in ('sale', 'done')
                GROUP BY
                    so.date_order::date, sol.product_id, pt.categ_id, so.warehouse_id, so.partner_id
            )
        """)

class SeasonalSalesAnalysis(models.Model):
    _name = 'seasonal.sales.analysis'
    _description = 'Seasonal Sales Analysis'
    _auto = False
    _order = 'avg_amount desc'

    season = fields.Selection([
        ('winter', 'Winter (Dec-Feb)'),
        ('spring', 'Spring (Mar-May)'),
        ('summer', 'Summer (Jun-Aug)'),
        ('fall', 'Fall (Sep-Nov)')
    ], string='Season')
    product_id = fields.Many2one('product.product', string='Product')
    category_id = fields.Many2one('product.category', string='Product Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    avg_quantity = fields.Float(string='Average Quantity Sold')
    avg_amount = fields.Float(string='Average Amount Sold')
    year = fields.Char(string='Year')
    total_sales = fields.Float(string='Total Sales')
    count_orders = fields.Integer(string='Order Count')

    def init(self):
        """Initialize the view"""
        try:
            self._cr.execute("DROP VIEW IF EXISTS seasonal_sales_analysis CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS seasonal_sales_analysis CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view with corrected GROUP BY clause
        self._cr.execute("""
            CREATE OR REPLACE VIEW seasonal_sales_analysis AS (
                SELECT
                    ROW_NUMBER() OVER() as id,
                    CASE
                        WHEN EXTRACT(MONTH FROM so.date_order) IN (12, 1, 2) THEN 'winter'
                        WHEN EXTRACT(MONTH FROM so.date_order) IN (3, 4, 5) THEN 'spring'
                        WHEN EXTRACT(MONTH FROM so.date_order) IN (6, 7, 8) THEN 'summer'
                        WHEN EXTRACT(MONTH FROM so.date_order) IN (9, 10, 11) THEN 'fall'
                    END as season,
                    EXTRACT(YEAR FROM so.date_order)::text as year,
                    sol.product_id as product_id,
                    pt.categ_id as category_id,
                    so.warehouse_id as warehouse_id,
                    AVG(sol.product_uom_qty) as avg_quantity,
                    AVG(sol.price_subtotal) as avg_amount,
                    SUM(sol.price_subtotal) as total_sales,
                    COUNT(DISTINCT so.id) as count_orders
                FROM
                    sale_order_line sol
                JOIN
                    sale_order so ON so.id = sol.order_id
                JOIN
                    product_product p ON p.id = sol.product_id
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                WHERE
                    so.state in ('sale', 'done')
                GROUP BY
                    season, EXTRACT(YEAR FROM so.date_order), sol.product_id, pt.categ_id, so.warehouse_id
            )
        """)

class DayOfWeekSalesAnalysis(models.Model):
    _name = 'day.of.week.sales.analysis'
    _description = 'Day of Week Sales Analysis'
    _auto = False
    _order = 'day_of_week'

    day_of_week = fields.Selection([
        ('0', 'Sunday'),
        ('1', 'Monday'),
        ('2', 'Tuesday'),
        ('3', 'Wednesday'),
        ('4', 'Thursday'),
        ('5', 'Friday'),
        ('6', 'Saturday')
    ], string='Day of Week')
    day_name = fields.Char(string='Day Name')
    product_id = fields.Many2one('product.product', string='Product')
    category_id = fields.Many2one('product.category', string='Product Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    avg_quantity = fields.Float(string='Average Quantity Sold')
    avg_amount = fields.Float(string='Average Amount Sold')
    total_orders = fields.Integer(string='Total Orders')
    total_sales = fields.Float(string='Total Sales')
    month = fields.Char(string='Month')
    year = fields.Char(string='Year')

    def init(self):
        """Initialize the view"""
        try:
            self._cr.execute("DROP VIEW IF EXISTS day_of_week_sales_analysis CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS day_of_week_sales_analysis CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view with corrected GROUP BY clause
        self._cr.execute("""
            CREATE OR REPLACE VIEW day_of_week_sales_analysis AS (
                SELECT
                    ROW_NUMBER() OVER() as id,
                    EXTRACT(DOW FROM so.date_order)::text as day_of_week,
                    CASE
                        WHEN EXTRACT(DOW FROM so.date_order) = 0 THEN 'Sunday'
                        WHEN EXTRACT(DOW FROM so.date_order) = 1 THEN 'Monday'
                        WHEN EXTRACT(DOW FROM so.date_order) = 2 THEN 'Tuesday'
                        WHEN EXTRACT(DOW FROM so.date_order) = 3 THEN 'Wednesday'
                        WHEN EXTRACT(DOW FROM so.date_order) = 4 THEN 'Thursday'
                        WHEN EXTRACT(DOW FROM so.date_order) = 5 THEN 'Friday'
                        WHEN EXTRACT(DOW FROM so.date_order) = 6 THEN 'Saturday'
                    END as day_name,
                    sol.product_id as product_id,
                    pt.categ_id as category_id,
                    so.warehouse_id as warehouse_id,
                    AVG(sol.product_uom_qty) as avg_quantity,
                    AVG(sol.price_subtotal) as avg_amount,
                    COUNT(DISTINCT so.id) as total_orders,
                    SUM(sol.price_subtotal) as total_sales,
                    TO_CHAR(so.date_order, 'Month') as month,
                    EXTRACT(YEAR FROM so.date_order)::text as year
                FROM
                    sale_order_line sol
                JOIN
                    sale_order so ON so.id = sol.order_id
                JOIN
                    product_product p ON p.id = sol.product_id
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                WHERE
                    so.state in ('sale', 'done')
                GROUP BY
                    EXTRACT(DOW FROM so.date_order), day_name, sol.product_id, pt.categ_id, so.warehouse_id, 
                    TO_CHAR(so.date_order, 'Month'), EXTRACT(YEAR FROM so.date_order)
            )
        """)

class HolidayData(models.Model):
    _name = 'holiday.data'
    _description = 'Holiday Data'

    name = fields.Char(string='Holiday Name', required=True)
    description = fields.Text(string='Description')
    date = fields.Date(string='Date', required=True)
    type = fields.Char(string='Type')
    country = fields.Char(string='Country')
    year = fields.Integer(string='Year')

    _sql_constraints = [
        ('unique_holiday_date', 'unique(date, name)', 'This holiday already exists for this date!')
    ]

class HolidaySalesAnalysis(models.Model):
    _name = 'holiday.sales.analysis'
    _description = 'Holiday Sales Analysis'
    _auto = False
    _order = 'date desc'

    date = fields.Date(string='Date')
    is_holiday = fields.Boolean(string='Is Holiday')
    holiday_name = fields.Char(string='Holiday Name')
    product_id = fields.Many2one('product.product', string='Product')
    category_id = fields.Many2one('product.category', string='Product Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    avg_quantity = fields.Float(string='Average Quantity Sold')
    avg_amount = fields.Float(string='Average Amount Sold')
    total_orders = fields.Integer(string='Total Orders')
    total_sales = fields.Float(string='Total Sales')
    year = fields.Char(string='Year')
    month = fields.Char(string='Month')

    def init(self):
        """Initialize the view"""
        try:
            self._cr.execute("DROP VIEW IF EXISTS holiday_sales_analysis CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS holiday_sales_analysis CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view with corrected GROUP BY clause
        self._cr.execute("""
            CREATE OR REPLACE VIEW holiday_sales_analysis AS (
                SELECT
                    ROW_NUMBER() OVER() as id,
                    so.date_order::date as date,
                    CASE WHEN hd.id IS NOT NULL THEN true ELSE false END as is_holiday,
                    hd.name as holiday_name,
                    sol.product_id as product_id,
                    pt.categ_id as category_id,
                    so.warehouse_id as warehouse_id,
                    AVG(sol.product_uom_qty) as avg_quantity,
                    AVG(sol.price_subtotal) as avg_amount,
                    COUNT(DISTINCT so.id) as total_orders,
                    SUM(sol.price_subtotal) as total_sales,
                    EXTRACT(YEAR FROM so.date_order)::text as year,
                    TO_CHAR(so.date_order, 'Month') as month
                FROM
                    sale_order_line sol
                JOIN
                    sale_order so ON so.id = sol.order_id
                JOIN
                    product_product p ON p.id = sol.product_id
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                LEFT JOIN
                    holiday_data hd ON hd.date = so.date_order::date
                WHERE
                    so.state in ('sale', 'done')
                GROUP BY
                    so.date_order::date, is_holiday, hd.name, sol.product_id, pt.categ_id, so.warehouse_id, 
                    EXTRACT(YEAR FROM so.date_order), TO_CHAR(so.date_order, 'Month')
            )
        """)

class HolidayComparisonAnalysis(models.Model):
    _name = 'holiday.comparison.analysis'
    _description = 'Holiday Comparison Analysis'
    _auto = False
    _order = 'is_holiday desc, avg_amount desc'

    is_holiday = fields.Boolean(string='Is Holiday')
    product_id = fields.Many2one('product.product', string='Product')
    category_id = fields.Many2one('product.category', string='Product Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    avg_quantity = fields.Float(string='Average Quantity Sold')
    avg_amount = fields.Float(string='Average Amount Sold')
    total_orders = fields.Integer(string='Total Orders')
    total_sales = fields.Float(string='Total Sales')
    order_count = fields.Integer(string='Order Count')
    year = fields.Char(string='Year')

    def init(self):
        """Initialize the view"""
        try:
            self._cr.execute("DROP VIEW IF EXISTS holiday_comparison_analysis CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS holiday_comparison_analysis CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view with corrected GROUP BY clause
        self._cr.execute("""
            CREATE OR REPLACE VIEW holiday_comparison_analysis AS (
                SELECT
                    ROW_NUMBER() OVER() as id,
                    CASE WHEN hd.id IS NOT NULL THEN true ELSE false END as is_holiday,
                    sol.product_id as product_id,
                    pt.categ_id as category_id,
                    so.warehouse_id as warehouse_id,
                    AVG(sol.product_uom_qty) as avg_quantity,
                    AVG(sol.price_subtotal) as avg_amount,
                    COUNT(DISTINCT so.id) as total_orders,
                    SUM(sol.price_subtotal) as total_sales,
                    COUNT(sol.id) as order_count,
                    EXTRACT(YEAR FROM so.date_order)::text as year
                FROM
                    sale_order_line sol
                JOIN
                    sale_order so ON so.id = sol.order_id
                JOIN
                    product_product p ON p.id = sol.product_id
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                LEFT JOIN
                    holiday_data hd ON hd.date = so.date_order::date
                WHERE
                    so.state in ('sale', 'done')
                GROUP BY
                    is_holiday, sol.product_id, pt.categ_id, so.warehouse_id, EXTRACT(YEAR FROM so.date_order)
            )
        """)

class SalesStockComparisonDashboard(models.Model):
    _name = 'sales.stock.comparison.dashboard'
    _description = 'Sales vs Stock Comparison Dashboard'
    _auto = False
    _order = 'sales_quantity desc'

    product_id = fields.Many2one('product.product', string='Product')
    name = fields.Char(string='Product Name')
    category_id = fields.Many2one('product.category', string='Product Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    sales_quantity = fields.Float(string='Quantity Sold')
    sales_amount = fields.Float(string='Amount Sold')
    stock_quantity = fields.Float(string='Current Stock')
    stock_status = fields.Selection([
        ('low', 'Low Stock'),
        ('normal', 'Normal Stock'),
        ('overstock', 'Overstock'),
        ('no_sales', 'No Sales')
    ], string='Stock Status')

    stock_icon = fields.Char(string='Indicator', compute='_compute_stock_icon', store=False)

    @api.depends('stock_status')
    def _compute_stock_icon(self):
        for rec in self:
            if rec.stock_status == 'low':
                rec.stock_icon = '<span style="color:red; font-size:20px;">●</span>'
            elif rec.stock_status == 'normal':
                rec.stock_icon = '<span style="color:green; font-size:20px;">●</span>'
            elif rec.stock_status == 'overstock':
                rec.stock_icon = '<span style="color:orange; font-size:20px;">●</span>'
            else:
                rec.stock_icon = '<span style="color:gray; font-size:20px;">●</span>'

    def init(self):
        self._cr.execute("DROP VIEW IF EXISTS sales_stock_comparison_dashboard CASCADE")
        self._cr.execute("""
            CREATE OR REPLACE VIEW sales_stock_comparison_dashboard AS (
                SELECT
                    ROW_NUMBER() OVER() AS id,
                    p.id as product_id,
                    pt.name as name,
                    pt.categ_id as category_id,
                    sl.warehouse_id as warehouse_id,
                    COALESCE(SUM(sol.product_uom_qty), 0) as sales_quantity,
                    COALESCE(SUM(sol.price_subtotal), 0) as sales_amount,
                    COALESCE(SUM(sq.quantity), 0) as stock_quantity,
                    CASE 
                        WHEN SUM(sol.product_uom_qty) IS NULL OR SUM(sol.product_uom_qty) = 0 THEN 'no_sales'
                        WHEN SUM(sq.quantity) < SUM(sol.product_uom_qty) * 0.5 THEN 'low'
                        WHEN SUM(sq.quantity) > SUM(sol.product_uom_qty) * 2 THEN 'overstock'
                        ELSE 'normal'
                    END as stock_status
                FROM
                    product_product p
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
                LEFT JOIN
                    sale_order_line sol ON sol.product_id = p.id
                LEFT JOIN
                    sale_order so ON so.id = sol.order_id AND so.state IN ('sale', 'done')
                LEFT JOIN
                    stock_quant sq ON sq.product_id = p.id
                LEFT JOIN
                    stock_location sl ON sl.id = sq.location_id AND sl.usage = 'internal'
                GROUP BY
                    p.id, pt.name, pt.categ_id, sl.warehouse_id
            )
        """)

class SalesDashboardView(models.Model):
    _name = 'sales.dashboard.view'
    _description = 'Sales Dashboard View'
    _auto = False

    name = fields.Char(string='Dashboard Name', default='Sales Dashboard')

    def init(self):
        # This is a SQL view initialization
        query = """
        SELECT 1 as id, 'Sales Dashboard' as name
        """
        tools.drop_view_if_exists(self.env.cr, self._table)
        self.env.cr.execute(f"""CREATE or REPLACE VIEW {self._table} as ({query})""")

    @api.model
    def get_dashboard_summary(self):
        """Get summary data for the dashboard"""
        sales_data = self.env['sales.dashboard'].search([])

        return {
            'total_sales': sum(sales_data.mapped('amount_sold')),
            'total_quantity': sum(sales_data.mapped('qty_sold')),
            'avg_order_value': sum(sales_data.mapped('amount_sold')) / len(sales_data) if sales_data else 0,
        }

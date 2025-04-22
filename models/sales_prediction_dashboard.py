# -*- coding: utf-8 -*-
from odoo import models, fields, api, _
from datetime import datetime, timedelta
import logging
import psycopg2

_logger = logging.getLogger(__name__)


class SalesPredictionDashboard(models.Model):
    _name = 'sales.prediction.dashboard'
    _description = 'Sales Prediction Dashboard'
    _auto = False
    _order = 'date desc'

    name = fields.Char(string='Name')
    date = fields.Date(string='Date')
    product_id = fields.Many2one('product.product', string='Product')
    category_id = fields.Many2one('product.category', string='Product Category')
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse')
    prediction_value = fields.Float(string='Predicted Sales')
    prediction_period = fields.Selection([
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('quarterly', 'Quarterly'),
        ('yearly', 'Yearly'),
    ], string='Prediction Period')
    model_id = fields.Many2one('sales.prediction.model', string='Model')
    batch_id = fields.Many2one('sales.prediction.batch', string='Batch')
    state = fields.Selection([
        ('draft', 'Draft'),
        ('confirmed', 'Confirmed'),
        ('cancelled', 'Cancelled')
    ], string='Status')
    current_stock = fields.Float(string='Current Stock')

    def init(self):
        """Initialize the view"""
        try:
            self._cr.execute("DROP VIEW IF EXISTS sales_prediction_dashboard CASCADE")
        except psycopg2.Error:
            pass

        try:
            self._cr.execute("DROP TABLE IF EXISTS sales_prediction_dashboard CASCADE")
        except psycopg2.Error:
            pass

        # Now create the view
        self._cr.execute("""
            CREATE OR REPLACE VIEW sales_prediction_dashboard AS (
                SELECT
                    sp.id as id,
                    sp.name as name,
                    sp.date as date,
                    sp.product_id as product_id,
                    pt.categ_id as category_id,
                    sp.warehouse_id as warehouse_id,
                    sp.prediction_value as prediction_value,
                    sp.prediction_period as prediction_period,
                    sp.model_id as model_id,
                    sp.batch_id as batch_id,
                    sp.state as state,
                    COALESCE(sq.quantity, 0) as current_stock
                FROM
                    sales_prediction sp
                JOIN
                    product_product p ON p.id = sp.product_id
                JOIN
                    product_template pt ON pt.id = p.product_tmpl_id
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
                    ) sq ON sq.product_id = sp.product_id
                WHERE
                    sp.state in ('draft', 'confirmed')
            )
        """)
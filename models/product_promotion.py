from odoo import models, fields, api
from datetime import timedelta


class ProductPromotion(models.Model):
    _name = 'product.promotion'
    _description = 'Product Promotion'
    _order = 'date_end'

    name = fields.Char(related='product_id.display_name', string='Product Name', store=True)
    product_id = fields.Many2one('product.product', string='Product', required=True, index=True)
    product_image = fields.Binary(related='product_id.image_128', string='Image')
    list_price = fields.Float(related='product_id.list_price', string='Regular Price')
    promotion_price = fields.Float(string='Promotion Price')
    discount_percentage = fields.Float(string='Discount %', compute='_compute_discount', store=True)
    date_start = fields.Date(string='Start Date')
    date_end = fields.Date(string='End Date')
    pricelist_item_id = fields.Many2one('product.pricelist.item', string='Pricelist Item')
    active = fields.Boolean(default=True)

    @api.depends('list_price', 'promotion_price')
    def _compute_discount(self):
        for promo in self:
            if promo.list_price and promo.promotion_price and promo.list_price > 0:
                promo.discount_percentage = ((promo.list_price - promo.promotion_price) / promo.list_price) * 100
            else:
                promo.discount_percentage = 0.0

    @api.model
    def update_promotion_products(self):
        """Update the list of products on promotion"""
        # Clear old records that are no longer valid
        expired_promos = self.search([
            '|',
            ('date_end', '<', fields.Date.today()),
            ('pricelist_item_id.date_end', '<', fields.Date.today())
        ])
        expired_promos.unlink()

        # Find products with price_extra
        products_with_extra = self.env['product.product'].search([
            ('price_extra', '>', 0)
        ])

        # Find products in active pricelist items with discounts
        pricelist_items = self.env['product.pricelist.item'].search([
            ('date_end', '>=', fields.Date.today()),
            ('applied_on', 'in', ['0_product_variant', '1_product']),
            '|',
            ('compute_price', '=', 'fixed'),
            ('compute_price', '=', 'percentage')
        ])

        # Process pricelist items
        for item in pricelist_items:
            if item.applied_on == '0_product_variant' and item.product_id:
                product = item.product_id
                # Check if this promotion already exists
                existing = self.search([
                    ('product_id', '=', product.id),
                    ('pricelist_item_id', '=', item.id)
                ])

                if not existing:
                    # Calculate promotion price
                    if item.compute_price == 'fixed':
                        promo_price = item.fixed_price
                    elif item.compute_price == 'percentage':
                        promo_price = product.list_price * (1 - (item.percent_price / 100))
                    else:
                        continue

                    # Create promotion record
                    self.create({
                        'product_id': product.id,
                        'promotion_price': promo_price,
                        'date_start': item.date_start,
                        'date_end': item.date_end,
                        'pricelist_item_id': item.id,
                    })

            elif item.applied_on == '1_product' and item.product_tmpl_id:
                # Get all variants of this template
                products = self.env['product.product'].search([
                    ('product_tmpl_id', '=', item.product_tmpl_id.id)
                ])

                for product in products:
                    # Check if this promotion already exists
                    existing = self.search([
                        ('product_id', '=', product.id),
                        ('pricelist_item_id', '=', item.id)
                    ])

                    if not existing:
                        # Calculate promotion price
                        if item.compute_price == 'fixed':
                            promo_price = item.fixed_price
                        elif item.compute_price == 'percentage':
                            promo_price = product.list_price * (1 - (item.percent_price / 100))
                        else:
                            continue

                        # Create promotion record
                        self.create({
                            'product_id': product.id,
                            'promotion_price': promo_price,
                            'date_start': item.date_start,
                            'date_end': item.date_end,
                            'pricelist_item_id': item.id,
                        })

        # Process products with price_extra
        for product in products_with_extra:
            # Check if this promotion already exists
            existing = self.search([
                ('product_id', '=', product.id),
                ('pricelist_item_id', '=', False)
            ])


        return True

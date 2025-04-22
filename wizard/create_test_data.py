from odoo import models, fields, api
import random
from datetime import timedelta
from faker import Faker


class CreateTestDataWizard(models.TransientModel):
    _name = 'create.test.data.wizard'
    _description = 'Generate realistic customer test data'

    customer_count = fields.Integer(default=10, required=True)
    orders_per_customer = fields.Integer(default=3, required=True)
    products_per_order = fields.Integer(default=2, required=True)
    max_days_ago = fields.Integer(
        string='Oldest Order (days)',
        default=365,
        help="Maximum days in the past for first orders"
    )

    def action_create_test_data(self):
        self.ensure_one()
        fake = Faker()

        # Prepare products
        products = self.env['product.product'].search([('sale_ok', '=', True)])
        if len(products) < 5:
            products = self._create_sample_products()

        # Create customers with orders
        for _ in range(self.customer_count):
            partner = self._create_customer(fake)
            self._create_orders_for_partner(partner, products)

        # Recompute all metrics
        self.env['res.partner'].search([])._compute_customer_metrics()

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Success',
                'message': f'Created {self.customer_count} customers with orders',
                'type': 'success'
            }
        }

    def _create_sample_products(self):
        products = self.env['product.product']
        for i in range(5):
            products |= products.create({
                'name': f'Sample Product {i + 1}',
                'type': 'product',
                'list_price': random.uniform(10, 100),
                'sale_ok': True
            })
        return products

    def _create_customer(self, fake):
        return self.env['res.partner'].create({
            'name': fake.name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'street': fake.street_address(),
            'city': fake.city(),
            'country_id': self.env.ref('base.us').id,
            'customer_rank': 1
        })

    def _create_orders_for_partner(self, partner, products):
        first_order_days_ago = random.randint(30, self.max_days_ago)
        order_dates = [
            fields.Datetime.now() - timedelta(days=first_order_days_ago - i * 30)
            for i in range(self.orders_per_customer)
        ]

        for date in sorted(order_dates):
            order = self.env['sale.order'].create({
                'partner_id': partner.id,
                'date_order': date,
                'state': 'sale'
            })

            selected_products = random.sample(products.ids,
                                              min(self.products_per_order, len(products)))

            for product_id in selected_products:
                self.env['sale.order.line'].create({
                    'order_id': order.id,
                    'product_id': product_id,
                    'product_uom_qty': random.randint(1, 5),
                    'price_unit': products.browse(product_id).list_price
                })
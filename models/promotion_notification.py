# -*- coding: utf-8 -*-
from odoo import models, fields, api
from datetime import datetime
import logging

_logger = logging.getLogger(__name__)


class PromotionNotification(models.Model):
    _name = 'promotion.notification'
    _description = 'Promotion Notification'

    name = fields.Char(string='Name', required=True)
    date_sent = fields.Datetime(string='Date Sent')
    product_count = fields.Integer(string='Products on Promotion', readonly=True)
    customer_count = fields.Integer(string='Customers Notified', readonly=True)

    @api.model
    def notify_customers_about_promotions(self):
        """
        Cette méthode est appelée par le cron :
        1. Récupère les produits en promotion
        2. Récupère les segments avec des recommandations vers ces produits
        3. Envoie les emails aux clients de ces segments
        """
        _logger.info("Starting notify_customers_about_promotions")

        # Étape 1 : Met à jour la liste des promotions
        self.env['product.promotion'].update_promotion_products()
        _logger.info("Updated promotion products")

        # Étape 2 : Récupère les promotions actives
        promotions = self.env['product.promotion'].search([
            ('date_end', '>=', fields.Date.today())
        ])
        _logger.info(f"Found {len(promotions)} active promotions")

        if not promotions:
            return False

        promotion_products = promotions.mapped('product_id')
        _logger.info(f"Found {len(promotion_products)} products on promotion")

        # Création d'un enregistrement de notification
        notification = self.create({
            'name': f'Promotion Notification {fields.Date.today()}',
            'date_sent': fields.Datetime.now(),
            'product_count': len(promotion_products),
            'customer_count': 0,
        })

        # Étape 3 : Recommandations de produits vers les segments
        recommendations = self.env['product.recommendation'].search([
            ('product_id', 'in', promotion_products.ids),
            ('score', '>=', 0.5)  # Seuil de recommandation
        ])
        _logger.info(f"Found {len(recommendations)} recommendations")

        segments = recommendations.mapped('segment_id')
        _logger.info(f"Found {len(segments)} customer segments")

        # Vérifier la configuration de l'email de l'entreprise
        company_email = self.env.company.email
        if not company_email:
            _logger.error("Company email not configured. Please set up company email in Settings > Companies")
            return False

        _logger.info(f"Using company email: {company_email}")

        # Obtenir l'utilisateur admin pour l'envoi des emails
        admin_user = self.env.ref('base.user_admin')

        # Nombre total de clients notifiés
        customers_notified = 0

        # Envoi des emails segment par segment
        for segment in segments:
            # Filtrer les clients avec des emails valides
            customers = segment.partner_ids.filtered(lambda p: p.email and '@' in p.email)
            _logger.info(f"Segment {segment.name}: Found {len(customers)} customers with valid email")

            if not customers:
                continue

            segment_recommendations = recommendations.filtered(lambda r: r.segment_id == segment)
            segment_products = segment_recommendations.mapped('product_id')

            # Préparer le contenu de l'email
            product_table = ""
            currency_symbol = self.env.company.currency_id.symbol

            # Get all promotions for the segment products
            product_promotions = {}
            for promotion in promotions:
                if promotion.product_id in segment_products:
                    product_promotions[promotion.product_id.id] = promotion

            for product in segment_products:
                promotion = product_promotions.get(product.id)
                if promotion:
                    regular_price = promotion.list_price
                    promo_price = promotion.promotion_price
                    discount = promotion.discount_percentage
                    start_date = promotion.date_start.strftime('%d/%m/%Y') if promotion.date_start else 'N/A'
                    end_date = promotion.date_end.strftime('%d/%m/%Y') if promotion.date_end else 'N/A'

                    # Format the product row with promotion details
                    product_table += f"""
                    <tr style="border-bottom: 1px solid #e6e9ef;">
                        <td style="padding: 15px; color: #333333; font-size: 15px;">{product.name}</td>
                        <td style="padding: 15px; color: #555555; font-size: 14px;">{product.description_sale or ''}</td>
                        <td style="padding: 15px; text-align: right; color: #888888; font-size: 14px; text-decoration: line-through;">
                            {currency_symbol} {regular_price:.2f}
                        </td>
                        <td style="padding: 15px; text-align: right; color: #e63946; font-weight: 600; font-size: 15px;">
                            {currency_symbol} {promo_price:.2f}
                        </td>
                        <td style="padding: 15px; text-align: center; color: #ffffff; font-weight: 600; font-size: 14px;">
                            <span style="background-color: #e63946; padding: 5px 8px; border-radius: 4px;">
                                -{discount:.0f}%
                            </span>
                        </td>
                        <td style="padding: 15px; text-align: center; color: #555555; font-size: 13px;">
                            {start_date} - {end_date}
                        </td>
                    </tr>
                    """
                else:
                    # If no promotion found for this product, display regular info
                    product_table += f"""
                    <tr style="border-bottom: 1px solid #e6e9ef;">
                        <td style="padding: 15px; color: #333333; font-size: 15px;">{product.name}</td>
                        <td style="padding: 15px; color: #555555; font-size: 14px;">{product.description_sale or ''}</td>
                        <td style="padding: 15px; text-align: right; color: #3a5998; font-weight: 600; font-size: 15px;" colspan="4">
                            {currency_symbol} {product.lst_price:.2f}
                        </td>
                    </tr>
                    """

            # Envoyer les emails directement sans utiliser le template
            for customer in customers:
                try:
                    # Construire l'email HTML directement
                    body_html = f"""
                    <!-- Optimized HTML Email Template -->
<div style="margin: 0; padding: 0; background-color: #f8f9fc; font-family: 'Helvetica Neue', Arial, sans-serif;">
    <!-- Header -->
    <div style="max-width: 800px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <!-- Top Banner -->
        <div style="background: linear-gradient(135deg, #3a5998 0%, #5b72a9 100%); padding: 30px 20px; text-align: center;">
            <h1 style="color: #ffffff; margin: 0; font-weight: 600; font-size: 26px;">Special Offers For You</h1>
            <p style="color: #ffffff; margin: 10px 0 0; font-size: 16px;">Discover our exclusive promotions</p>
        </div>

        <!-- Main Content -->
        <div style="padding: 30px 25px;">
            <h2 style="color: #3a5998; font-weight: 500; margin-top: 0;">Hello {customer.name},</h2>

            <p style="color: #555555; line-height: 1.6; font-size: 16px; margin-bottom: 25px;">
                We've selected special promotions that might interest you! Don't miss these limited-time offers.
            </p>

            <h3 style="color: #3a5998; font-weight: 500; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e6e9ef;">
                Products on Promotion
            </h3>

            <!-- Products Table -->
            <table style="width: 100%; border-collapse: collapse; border-radius: 6px; overflow: hidden; margin-bottom: 30px;">
                <tr style="background-color: #3a5998; color: white;">
                    <th style="padding: 12px 15px; text-align: left; font-weight: 500; font-size: 15px;">Product</th>
                    <th style="padding: 12px 15px; text-align: left; font-weight: 500; font-size: 15px;">Description</th>
                    <th style="padding: 12px 15px; text-align: right; font-weight: 500; font-size: 15px;">Regular Price</th>
                    <th style="padding: 12px 15px; text-align: right; font-weight: 500; font-size: 15px;">Promo Price</th>
                    <th style="padding: 12px 15px; text-align: center; font-weight: 500; font-size: 15px;">Discount</th>
                    <th style="padding: 12px 15px; text-align: center; font-weight: 500; font-size: 15px;">Valid Period</th>
                </tr>

                <!-- Products Loop -->
                {product_table}
            </table>

            <!-- Call to Action Button -->
            <div style="text-align: center; margin: 35px 0;">
                <a href="/shop" style="background-color: #3a5998; color: white; padding: 12px 28px; text-decoration: none; border-radius: 30px; font-weight: 500; font-size: 16px; display: inline-block; transition: all 0.3s ease;">
                    Shop Now
                </a>
            </div>

            <p style="color: #555555; line-height: 1.6; font-size: 14px; margin-top: 25px; font-style: italic;">
                * Promotions are valid for the specified dates only. Prices and availability may change without notice.
            </p>
        </div>

        <!-- Footer -->
        <div style="background-color: #f3f5f9; padding: 25px; text-align: center;">
            <p style="color: #777777; font-size: 14px; margin: 0 0 10px;">
                Thank you for your loyalty!
            </p>
            <p style="color: #999999; font-size: 12px; margin: 15px 0 0;">
                If you no longer wish to receive these emails, please contact our customer service.
            </p>

            <!-- Social Media Links -->
            <div style="margin-top: 20px;">
                <a href="#" style="display: inline-block; margin: 0 8px; color: #3a5998; text-decoration: none;">
                    <span style="font-size: 22px;">&#xf09a;</span>
                </a>
                <a href="#" style="display: inline-block; margin: 0 8px; color: #3a5998; text-decoration: none;">
                    <span style="font-size: 22px;">&#xf099;</span>
                </a>
                <a href="#" style="display: inline-block; margin: 0 8px; color: #3a5998; text-decoration: none;">
                    <span style="font-size: 22px;">&#xf16d;</span>
                </a>
            </div>
        </div>
    </div>
</div>
                    """

                    # Créer les valeurs de l'email
                    mail_values = {
                        'subject': '🔥Special Promotions Just For You!',
                        'body_html': body_html,
                        'email_from': company_email,  # Utiliser directement l'email de la société
                        'email_to': customer.email,
                        'author_id': admin_user.partner_id.id,  # Définir l'auteur comme admin
                        'recipient_ids': [(6, 0, [customer.id])],  # Ajouter le destinataire explicitement
                        'auto_delete': True,
                    }

                    _logger.info(f"Sending email to {customer.email} with {len(segment_products)} products")

                    # Créer et envoyer l'email
                    mail = self.env['mail.mail'].sudo().create(mail_values)

                    # Vérifier si l'email a été créé correctement
                    if mail and mail.id:
                        # Envoyer l'email immédiatement
                        mail_sent = mail.sudo().send(raise_exception=False)

                        if mail_sent:
                            _logger.info(f"Email successfully sent to {customer.email}")
                            customers_notified += 1
                        else:
                            _logger.warning(f"Failed to send email to {customer.email}")
                            # Vérifier pourquoi l'envoi a échoué
                            if mail.exists():
                                _logger.warning(f"Mail state: {mail.state}, Failure reason: {mail.failure_reason}")
                    else:
                        _logger.error(f"Failed to create mail for {customer.email}")

                except Exception as e:
                    _logger.error(f"Error sending email to {customer.email}: {str(e)}")

        # Mise à jour de l'enregistrement de notification
        notification.write({
            'customer_count': customers_notified or sum(
                len(segment.partner_ids.filtered(lambda p: p.email and '@' in p.email)) for segment in segments)
        })
        _logger.info(f"Notification complete. Notified {customers_notified} customers")
        return True

    def test_email_sending(self):
        """
        Méthode de test pour vérifier l'envoi d'un email
        """
        try:
            # Vérifier la configuration de l'email
            company_email = self.env.company.email
            if not company_email:
                _logger.error("Company email not configured")
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Error',
                        'message': 'Company email not configured. Please set it in company settings.',
                        'type': 'danger',
                    }
                }

            # Utiliser l'email de l'utilisateur actuel comme destinataire de test
            test_email = self.env.user.email
            if not test_email:
                _logger.error("Current user has no email configured")
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Error',
                        'message': 'Current user has no email configured.',
                        'type': 'danger',
                    }
                }

            # Vérifier les serveurs de messagerie
            mail_servers = self.env['ir.mail_server'].search([])
            if not mail_servers:
                _logger.warning("No mail servers configured")
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Warning',
                        'message': 'No mail servers configured. Email might not be sent.',
                        'type': 'warning',
                    }
                }

            _logger.info(f"Preparing test email from {company_email} to {test_email}")

            # Create a test promotion for demonstration
            test_product = self.env['product.product'].search([], limit=1)
            if test_product:
                # Create sample promotion data for the test email
                promotion_data = {
                    'regular_price': test_product.lst_price,
                    'promo_price': test_product.lst_price * 0.8,  # 20% off
                    'discount': 20,
                    'start_date': fields.Date.today().strftime('%d/%m/%Y'),
                    'end_date': (fields.Date.today() + fields.timedelta(days=30)).strftime('%d/%m/%Y')
                }

                currency_symbol = self.env.company.currency_id.symbol

                # Sample product row with promotion details
                product_row = f"""
                <tr style="border-bottom: 1px solid #e6e9ef;">
                    <td style="padding: 15px; color: #333333; font-size: 15px;">{test_product.name}</td>
                    <td style="padding: 15px; color: #555555; font-size: 14px;">{test_product.description_sale or 'Sample product description'}</td>
                    <td style="padding: 15px; text-align: right; color: #888888; font-size: 14px; text-decoration: line-through;">
                        {currency_symbol} {promotion_data['regular_price']:.2f}
                    </td>
                    <td style="padding: 15px; text-align: right; color: #e63946; font-weight: 600; font-size: 15px;">
                        {currency_symbol} {promotion_data['promo_price']:.2f}
                    </td>
                    <td style="padding: 15px; text-align: center; color: #ffffff; font-weight: 600; font-size: 14px;">
                        <span style="background-color: #e63946; padding: 5px 8px; border-radius: 4px;">
                            -{promotion_data['discount']:.0f}%
                        </span>
                    </td>
                    <td style="padding: 15px; text-align: center; color: #555555; font-size: 13px;">
                        {promotion_data['start_date']} - {promotion_data['end_date']}
                    </td>
                </tr>
                """

                # Créer un email de test avec le format de promotion
                body_html = f"""
                <!-- Optimized HTML Email Template -->
                <div style="margin: 0; padding: 0; background-color: #f8f9fc; font-family: 'Helvetica Neue', Arial, sans-serif;">
                    <!-- Header -->
                    <div style="max-width: 800px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <!-- Top Banner -->
                        <div style="background: linear-gradient(135deg, #3a5998 0%, #5b72a9 100%); padding: 30px 20px; text-align: center;">
                            <h1 style="color: #ffffff; margin: 0; font-weight: 600; font-size: 26px;">Test Email - Promotion Format</h1>
                            <p style="color: #ffffff; margin: 10px 0 0; font-size: 16px;">This is a test of the promotion email template</p>
                        </div>

                        <!-- Main Content -->
                        <div style="padding: 30px 25px;">
                            <h2 style="color: #3a5998; font-weight: 500; margin-top: 0;">Hello {self.env.user.name},</h2>

                            <p style="color: #555555; line-height: 1.6; font-size: 16px; margin-bottom: 25px;">
                                This is a test email to verify the promotion notification format. If you received this email, your email configuration is working correctly.
                            </p>

                            <h3 style="color: #3a5998; font-weight: 500; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e6e9ef;">
                                Sample Promotion
                            </h3>

                            <!-- Products Table -->
                            <table style="width: 100%; border-collapse: collapse; border-radius: 6px; overflow: hidden; margin-bottom: 30px;">
                                <tr style="background-color: #3a5998; color: white;">
                                    <th style="padding: 12px 15px; text-align: left; font-weight: 500; font-size: 15px;">Product</th>
                                    <th style="padding: 12px 15px; text-align: left; font-weight: 500; font-size: 15px;">Description</th>
                                    <th style="padding: 12px 15px; text-align: right; font-weight: 500; font-size: 15px;">Regular Price</th>
                                    <th style="padding: 12px 15px; text-align: right; font-weight: 500; font-size: 15px;">Promo Price</th>
                                    <th style="padding: 12px 15px; text-align: center; font-weight: 500; font-size: 15px;">Discount</th>
                                    <th style="padding: 12px 15px; text-align: center; font-weight: 500; font-size: 15px;">Valid Period</th>
                                </tr>

                                <!-- Sample Product -->
                                {product_row}
                            </table>

                            <p style="color: #555555; line-height: 1.6; font-size: 14px;">
                                Time: {fields.Datetime.now()}
                            </p>
                        </div>

                        <!-- Footer -->
                        <div style="background-color: #f3f5f9; padding: 25px; text-align: center;">
                            <p style="color: #777777; font-size: 14px; margin: 0 0 10px;">
                                This is a test email
                            </p>
                        </div>
                    </div>
                </div>
                """
            else:
                # Fallback to simple test email if no products found
                body_html = f"""
                <div style="margin: 0px; padding: 0px; font-family: Arial, Helvetica, sans-serif;">
                    <h2 style="color: #7C7BAD; font-weight: bold;">Test Email</h2>
                    <p>This is a test email from Odoo.</p>
                    <p>If you received this email, your email configuration is working correctly.</p>
                    <p>Time: {fields.Datetime.now()}</p>
                </div>
                """

            # Créer les valeurs de l'email
            mail_values = {
                'subject': 'Test Email from Odoo - Promotion Template',
                'body_html': body_html,
                'email_from': company_email,
                'email_to': test_email,
                'author_id': self.env.user.partner_id.id,
                'recipient_ids': [(6, 0, [self.env.user.partner_id.id])],
                'auto_delete': False,  # Conserver l'email pour le débogage
            }

            # Créer et envoyer l'email
            _logger.info(f"Creating mail with values: {mail_values}")
            mail = self.env['mail.mail'].sudo().create(mail_values)
            _logger.info(f"Mail created with ID: {mail.id}")

            # Envoyer l'email immédiatement
            result = mail.sudo().send(raise_exception=False)
            _logger.info(f"Mail send result: {result}")

            if result:
                _logger.info("Test email sent successfully")
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Success',
                        'message': f'Test email sent successfully to {test_email}',
                        'type': 'success',
                    }
                }
            else:
                _logger.error("Test email failed to send")
                # Vérifier l'état du mail
                if mail.exists():
                    _logger.error(f"Mail state: {mail.state}")
                    _logger.error(f"Mail failure reason: {mail.failure_reason}")

                    return {
                        'type': 'ir.actions.client',
                        'tag': 'display_notification',
                        'params': {
                            'title': 'Error',
                            'message': f'Failed to send test email. Reason: {mail.failure_reason or "Unknown"}',
                            'type': 'danger',
                        }
                    }
                else:
                    return {
                        'type': 'ir.actions.client',
                        'tag': 'display_notification',
                        'params': {
                            'title': 'Error',
                            'message': 'Failed to send test email. Mail record was deleted.',
                            'type': 'danger',
                        }
                    }

        except Exception as e:
            _logger.error(f"Test email error: {str(e)}")
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Error',
                    'message': f'Test email error: {str(e)}',
                    'type': 'danger',
                }
            }

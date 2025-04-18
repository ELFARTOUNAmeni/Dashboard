# -*- coding: utf-8 -*-
from odoo import models, fields, api, _
import logging
import math

_logger = logging.getLogger(__name__)


class DemandPredictionAlert(models.Model):
    _name = 'demand.prediction.alert'
    _description = 'Demand Prediction Alert'
    _order = 'severity desc, id desc'
    _inherit = ['mail.thread', 'mail.activity.mixin']

    name = fields.Char('Alert', required=True, tracking=True)
    prediction_id = fields.Many2one('demand.prediction', string='Prediction', required=True, index=True, tracking=True ,ondelete='cascade')
    product_id = fields.Many2one('product.product', string='Product', related='prediction_id.product_id', store=True,
                                 index=True)
    warehouse_id = fields.Many2one('stock.warehouse', string='Warehouse', related='prediction_id.warehouse_id',
                                   store=True, index=True)

    alert_type = fields.Selection([
        ('low_stock', 'Low Stock'),
        ('high_demand', 'High Demand'),
        ('stockout_risk', 'Stockout Risk'),
        ('high_stock', 'High Stock')
    ], string='Type', required=True, tracking=True, index=True)

    severity = fields.Selection([
        ('0', 'Normal'),
        ('1', 'Low'),
        ('2', 'Medium'),
        ('3', 'High'),
        ('4', 'Critical')
    ], string='Severity', default='0', required=True, tracking=True, index=True)

    stock_available = fields.Float('Current Stock', tracking=True)
    date = fields.Datetime('Date', default=fields.Datetime.now, tracking=True, index=True)
    state = fields.Selection([
        ('new', 'New'),
        ('acknowledged', 'Acknowledged'),
        ('resolved', 'Resolved')
    ], string='Status', default='new', tracking=True, index=True)

    user_id = fields.Many2one('res.users', string='User', default=lambda self: self.env.user, tracking=True)
    company_id = fields.Many2one('res.company', string='Company', default=lambda self: self.env.company, tracking=True)

    action_history_ids = fields.One2many('demand.prediction.alert.action', 'alert_id', string='Action History')
    recommended_action = fields.Text('Recommended Action', compute='_compute_recommended_action')

    @api.depends('alert_type', 'severity', 'stock_available', 'prediction_id.prediction_value')
    def _compute_recommended_action(self):
        for alert in self:
            try:
                predicted = alert.prediction_id.prediction_value or 0
                current_stock = alert.stock_available or 0
                coverage = (current_stock / predicted * 100) if predicted else 0
                shortage = predicted - current_stock

                if alert.alert_type == 'low_stock':
                    if alert.severity in ['4', '3']:
                        alert.recommended_action = _(
                            "Urgent: Place purchase order for at least %.2f units. "
                            "Current stock covers only %.1f%% of predicted demand."
                        ) % (shortage, coverage)
                    else:
                        alert.recommended_action = _(
                            "Consider placing purchase order for %.2f units to maintain optimal stock levels."
                        ) % shortage

                elif alert.alert_type == 'stockout_risk':
                    alert.recommended_action = _(
                        "CRITICAL: Immediate action required. Place emergency order for %.2f units. "
                        "Contact suppliers to expedite delivery."
                    ) % shortage

                elif alert.alert_type == 'high_demand':
                    alert.recommended_action = _(
                        "Unusual demand detected. Verify sales forecasts and consider increasing safety stock by 20%%."
                    )

                elif alert.alert_type == 'high_stock':
                    excess = current_stock - predicted
                    alert.recommended_action = _(
                        "Excess inventory detected. Consider reducing future orders or running promotions. "
                        "Current stock exceeds predicted demand by %.2f units."
                    ) % excess

                else:
                    alert.recommended_action = _("Review inventory levels and take appropriate action.")
            except Exception:
                alert.recommended_action = _("Unable to calculate recommendation.")

    @api.model
    def create_alert(self, prediction, alert_type, stock, severity=None):
        severities = {
            'low_stock': '2',
            'high_demand': '2',
            'stockout_risk': '3',
            'high_stock': '1',
        }

        severity_str = str(severity) if severity else severities.get(alert_type, '1')

        # Récupérer le stock actuel directement depuis le produit
        actual_stock = prediction.product_id.with_context(warehouse=prediction.warehouse_id.id).qty_available

        existing_alert = self.search([
            ('prediction_id', '=', prediction.id),
            ('alert_type', '=', alert_type),
            ('state', 'in', ['new', 'acknowledged'])
        ], limit=1)

        if existing_alert:
            existing_alert.write({
                'severity': max(existing_alert.severity, severity_str),
                'stock_available': actual_stock,  # Utiliser le stock actuel du produit
            })
            alert = existing_alert
        else:
            alert = self.create({
                'name': _(f"Alert: {alert_type.replace('_', ' ').title()} - {prediction.product_id.name}"),
                'prediction_id': prediction.id,
                'alert_type': alert_type,
                'severity': severity_str,
                'stock_available': actual_stock,  # Utiliser le stock actuel du produit
                'user_id': prediction.generator_id.id
            })

        # Send email notification using direct method
        self._send_direct_email(alert)
        return alert

    @api.model
    def _send_direct_email(self, alert):
        """Send email directly without using templates"""
        try:
            # Log the attempt
            _logger.info(f"Attempting to send notification email for alert {alert.id}")

            # Check if user has email
            if not alert.user_id.email:
                _logger.warning(f"User {alert.user_id.name} has no email configured")
                self.env['demand.prediction.log'].log_operation(
                    'email_notification',
                    'failed',
                    f"L'utilisateur {alert.user_id.name} n'a pas d'adresse email configurée",
                    0.0
                )
                return False

            # Format today's date in French
            today_date = fields.Date.today().strftime("%d %B %Y").replace(
                "January", "janvier").replace("February", "février").replace("March", "mars"
                                                                             ).replace("April", "avril").replace("May",
                                                                                                                 "mai").replace(
                "June", "juin"
            ).replace("July", "juillet").replace("August", "août").replace("September", "septembre"
                                                                           ).replace("October", "octobre").replace(
                "November", "novembre").replace("December", "décembre")

            # Get current year for copyright
            current_year = fields.Date.today().year

            # Get alert type display name
            alert_type_map = {
                'low_stock': 'Alerte de Stock Bas',
                'high_demand': 'Alerte de Demande Élevée',
                'stockout_risk': 'Risque de Rupture de Stock',
                'high_stock': 'Alerte de Stock Élevé'
            }
            alert_type_display = alert_type_map.get(alert.alert_type, alert.alert_type)

            # Modern color palette - Using vibrant, trendy colors
            primary_color = "#6366F1"  # Indigo
            secondary_color = "#EC4899"  # Pink
            success_color = "#10B981"  # Emerald
            warning_color = "#F59E0B"  # Amber
            danger_color = "#EF4444"  # Red
            info_color = "#3B82F6"  # Blue
            dark_color = "#1F2937"  # Gray 800
            light_color = "#F9FAFB"  # Gray 50
            muted_color = "#9CA3AF"  # Gray 400

            # Get alert icons - using modern SVG icons with vibrant colors
            alert_icon_map = {
                'low_stock': f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{danger_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
                'high_demand': f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{info_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>',
                'stockout_risk': f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{warning_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
                'high_stock': f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{success_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>'
            }
            alert_icon = alert_icon_map.get(alert.alert_type, '')

            # Get severity details with modern color palette
            severity_map = {
                '4': (
                    'CRITIQUE', danger_color,
                    f"rgba({int(danger_color[1:3], 16)}, {int(danger_color[3:5], 16)}, {int(danger_color[5:7], 16)}, 0.08)",
                    f"rgba({int(danger_color[1:3], 16)}, {int(danger_color[3:5], 16)}, {int(danger_color[5:7], 16)}, 0.15)",
                    f"rgba({int(danger_color[1:3], 16)}, {int(danger_color[3:5], 16)}, {int(danger_color[5:7], 16)}, 0.18)",
                    f"linear-gradient(135deg, {danger_color} 0%, #DC2626 100%)"),
                '3': (
                    'ÉLEVÉE', warning_color,
                    f"rgba({int(warning_color[1:3], 16)}, {int(warning_color[3:5], 16)}, {int(warning_color[5:7], 16)}, 0.08)",
                    f"rgba({int(warning_color[1:3], 16)}, {int(warning_color[3:5], 16)}, {int(warning_color[5:7], 16)}, 0.15)",
                    f"rgba({int(warning_color[1:3], 16)}, {int(warning_color[3:5], 16)}, {int(warning_color[5:7], 16)}, 0.18)",
                    f"linear-gradient(135deg, {warning_color} 0%, #D97706 100%)"),
                '2': (
                    'MOYENNE', info_color,
                    f"rgba({int(info_color[1:3], 16)}, {int(info_color[3:5], 16)}, {int(info_color[5:7], 16)}, 0.08)",
                    f"rgba({int(info_color[1:3], 16)}, {int(info_color[3:5], 16)}, {int(info_color[5:7], 16)}, 0.15)",
                    f"rgba({int(info_color[1:3], 16)}, {int(info_color[3:5], 16)}, {int(info_color[5:7], 16)}, 0.18)",
                    f"linear-gradient(135deg, {info_color} 0%, #2563EB 100%)"),
                '1': (
                    'BASSE', success_color,
                    f"rgba({int(success_color[1:3], 16)}, {int(success_color[3:5], 16)}, {int(success_color[5:7], 16)}, 0.08)",
                    f"rgba({int(success_color[1:3], 16)}, {int(success_color[3:5], 16)}, {int(success_color[5:7], 16)}, 0.15)",
                    f"rgba({int(success_color[1:3], 16)}, {int(success_color[3:5], 16)}, {int(success_color[5:7], 16)}, 0.18)",
                    f"linear-gradient(135deg, {success_color} 0%, #059669 100%)"),
                '0': (
                    'NORMALE', muted_color,
                    f"rgba({int(muted_color[1:3], 16)}, {int(muted_color[3:5], 16)}, {int(muted_color[5:7], 16)}, 0.08)",
                    f"rgba({int(muted_color[1:3], 16)}, {int(muted_color[3:5], 16)}, {int(muted_color[5:7], 16)}, 0.15)",
                    f"rgba({int(muted_color[1:3], 16)}, {int(muted_color[3:5], 16)}, {int(muted_color[5:7], 16)}, 0.18)",
                    f"linear-gradient(135deg, {muted_color} 0%, #6B7280 100%)")
            }

            # Default to NORMALE if severity not found
            severity_display, severity_color, bg_color, border_color, icon_bg_color, gradient = severity_map.get(
                alert.severity, ('NORMALE', muted_color,
                                 f"rgba({int(muted_color[1:3], 16)}, {int(muted_color[3:5], 16)}, {int(muted_color[5:7], 16)}, 0.05)",
                                 f"rgba({int(muted_color[1:3], 16)}, {int(muted_color[3:5], 16)}, {int(muted_color[5:7], 16)}, 0.1)",
                                 f"rgba({int(muted_color[1:3], 16)}, {int(muted_color[3:5], 16)}, {int(muted_color[5:7], 16)}, 0.12)",
                                 f"linear-gradient(135deg, {muted_color} 0%, #6B7280 100%)")
            )

            # Build ultra-modern email body with glassmorphism effects and improved typography
            body_html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Alerte de Prédiction</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    </head>
    <body style="margin: 0; padding: 0; background-color: #F3F4F6; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: {dark_color}; line-height: 1.5;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <!-- Main Card -->
            <div style="background-color: {light_color}; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.05); margin-bottom: 20px;">
                <!-- Header with gradient background -->
                <div style="background: {gradient}; padding: 40px 30px; text-align: center; position: relative;">
                    <!-- Glassmorphism logo container -->
                    <div style="width: 90px; height: 90px; background: rgba(255,255,255,0.25); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.3); box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                        <div style="font-weight: bold; font-size: 28px; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            {alert_icon}
                        </div>
                    </div>
                    <h1 style="color: white; margin: 0 0 5px; font-weight: 700; font-size: 24px; letter-spacing: -0.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        {alert_type_display}
                    </h1>
                    <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 16px; font-weight: 400;">
                        {alert.company_id.name}
                    </p>

                    <!-- Severity badge -->
                    <div style="position: absolute; top: 20px; right: 20px;">
                        <span style="display: inline-block; padding: 6px 16px; border-radius: 30px; font-weight: 600; font-size: 12px; letter-spacing: 0.5px; background-color: rgba(255,255,255,0.25); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); color: white; border: 1px solid rgba(255,255,255,0.3); box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            {severity_display}
                        </span>
                    </div>
                </div>

                <!-- Content -->
                <div style="padding: 30px;">
                    <!-- Greeting -->
                    <p style="margin-top: 0; font-size: 16px; color: {dark_color};">
                        Bonjour <strong style="font-weight: 600;">{alert.user_id.name}</strong>,
                    </p>

                    <p style="margin-bottom: 25px; color: {dark_color}; font-size: 16px; line-height: 1.6;">
                        Une nouvelle alerte a été générée par notre système de prédiction de demande et requiert votre attention.
                    </p>

                    <!-- Product and Warehouse Card -->
                    <div style="background-color: white; border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.05);">
                        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                            <div style="flex: 1; min-width: 200px;">
                                <p style="margin: 0 0 8px; font-size: 12px; color: {muted_color}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Produit</p>
                                <p style="margin: 0; font-size: 18px; font-weight: 600; color: {dark_color};">{alert.product_id.name}</p>
                            </div>
                            <div style="flex: 1; min-width: 200px;">
                                <p style="margin: 0 0 8px; font-size: 12px; color: {muted_color}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Entrepôt</p>
                                <p style="margin: 0; font-size: 18px; font-weight: 600; color: {dark_color};">{alert.warehouse_id.name}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Stock Info Card -->
                    <div style="background: linear-gradient(135deg, {light_color} 0%, white 100%); border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.05);">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <p style="margin: 0 0 8px; font-size: 12px; color: {muted_color}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Stock Actuel</p>
                                <p style="margin: 0; font-size: 28px; font-weight: 700; color: {dark_color}; letter-spacing: -0.5px;">
                                    {alert.stock_available} <span style="font-size: 14px; font-weight: normal; color: {muted_color};">unités</span>
                                </p>
                            </div>
                            <div style="width: 50px; height: 50px; background-color: {bg_color}; border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                                <span style="color: {severity_color}; font-size: 24px;">
                                    {alert_icon}
                                </span>
                            </div>
                        </div>
                    </div>

                    <!-- Date information -->
                    <div style="text-align: center; margin: 30px 0;">
                        <p style="display: inline-block; padding: 8px 20px; background-color: {light_color}; border-radius: 30px; font-size: 14px; color: {muted_color}; margin: 0; font-weight: 500;">
                            {today_date}
                        </p>
                    </div>

                    <!-- Recommended Action Section -->
                    <div style="margin: 30px 0;">
                        <h3 style="font-size: 18px; color: {dark_color}; margin: 0 0 15px 0; font-weight: 600; letter-spacing: -0.3px;">Action Recommandée</h3>
                        <div style="background-color: {bg_color}; border-radius: 12px; padding: 25px; border-left: 5px solid {severity_color};">
                            <p style="margin: 0; font-size: 15px; color: {dark_color}; line-height: 1.6;">
                                {alert.recommended_action or "Aucune action spécifique recommandée pour le moment. Veuillez surveiller l'évolution de cette alerte."}
                            </p>
                        </div>
                    </div>

                    <!-- CTA Button -->
                    <div style="text-align: center; margin-top: 35px;">
                        <a href="/web#id={alert.id}&amp;model=demand.prediction.alert&amp;view_type=form" 
                        style="display: inline-block; background: {gradient}; color: white; text-decoration: none; padding: 14px 36px; border-radius: 30px; font-weight: 600; font-size: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; letter-spacing: 0.3px;">
                            Consulter l'alerte
                        </a>
                        <p style="color: {muted_color}; font-size: 13px; margin-top: 12px;">
                            Pour une analyse plus détaillée et des options d'action
                        </p>
                    </div>
                </div>
            </div>

            <!-- Footer Card -->
            <div style="background-color: white; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
                <p style="margin: 0 0 10px 0; font-size: 14px; color: {dark_color};">Si vous avez des questions, contactez notre équipe support</p>
                <p style="margin: 0; font-size: 13px; color: {muted_color};">
                    &copy; {current_year} {alert.company_id.name} - Tous droits réservés
                </p>
            </div>
        </div>
    </body>
    </html>
    """
            # Emoji map for alert types
            alert_emoji_map = {
                'low_stock': '📉',
                'high_demand': '📈',
                'stockout_risk': '⚠️',
                'high_stock': '📊'
            }

            # Get emoji for alert type
            alert_emoji = alert_emoji_map.get(alert.alert_type, '⚠️')

            # Prepare email values with emoji in subject
            mail_values = {
                'subject': f"{alert_emoji} Alerte de prédiction: {alert.name}",
                'body_html': body_html,
                'email_from': alert.company_id.email or self.env.user.email_formatted,
                'email_to': alert.user_id.email,
                'auto_delete': True,
                'model': 'demand.prediction.alert',
                'res_id': alert.id,
            }

            # Send email directly using mail.mail
            mail = self.env['mail.mail'].sudo().create(mail_values)
            mail.send(raise_exception=True)

            # Log success
            self.env['demand.prediction.alert.action'].create({
                'alert_id': alert.id,
                'action_type': 'notification',
                'description': _('Email notification sent to %s') % alert.user_id.email,
                'user_id': self.env.user.id,
            })

            self.env['demand.prediction.log'].log_operation(
                'email_notification',
                'success',
                f"Email envoyé avec succès à {alert.user_id.email}",
                0.0
            )

            return True

        except Exception as e:
            # Log failure
            _logger.error(f"Error sending direct email: {str(e)}")
            self.env['demand.prediction.log'].log_operation(
                'email_notification',
                'failed',
                f"Erreur lors de l'envoi de l'email direct: {str(e)}",
                0.0
            )
            return False

    # Exemple d'utilisation (non inclus dans la fonction)
    # self._send_direct_email(alert)
    def action_acknowledge(self):
        for alert in self:
            alert.write({'state': 'acknowledged'})
            self.env['demand.prediction.alert.action'].create({
                'alert_id': alert.id,
                'action_type': 'status_change',
                'description': _('Alert acknowledged'),
                'user_id': self.env.user.id,
            })
        return True

    def action_resolve(self):
        for alert in self:
            alert.write({'state': 'resolved'})
            self.env['demand.prediction.alert.action'].create({
                'alert_id': alert.id,
                'action_type': 'status_change',
                'description': _('Alert resolved'),
                'user_id': self.env.user.id,
            })
        return True

    def action_refresh_stock(self):
        """Refresh the current stock level from the product"""
        for alert in self:
            actual_stock = alert.product_id.with_context(warehouse=alert.warehouse_id.id).qty_available
            alert.write({'stock_available': actual_stock})

            self.env['demand.prediction.alert.action'].create({
                'alert_id': alert.id,
                'action_type': 'note',
                'description': _('Stock level refreshed: %s units') % actual_stock,
                'user_id': self.env.user.id,
            })

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Stock Updated'),
                'message': _('Current stock levels have been refreshed'),
                'sticky': False,
                'type': 'success',
            }
        }

    def action_create_purchase_request(self):
        """Create a purchase request based on the alert information"""
        self.ensure_one()

        if not self.product_id or not self.warehouse_id:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Error'),
                    'message': _('Product or warehouse information missing'),
                    'sticky': False,
                    'type': 'danger',
                }
            }

        # Refresh stock level before creating purchase request
        actual_stock = self.product_id.with_context(warehouse=self.warehouse_id.id).qty_available
        self.write({'stock_available': actual_stock})

        # Calculate quantity to order based on alert type and prediction
        predicted_demand = self.prediction_id.prediction_value or 0
        current_stock = self.stock_available or 0

        if self.alert_type in ['low_stock', 'stockout_risk']:
            # For low stock or stockout risk, order the shortage plus safety margin
            shortage = max(0, predicted_demand - current_stock)
            safety_margin = 0.2  # 20% safety margin
            quantity = shortage * (1 + safety_margin)
        elif self.alert_type == 'high_demand':
            # For high demand, order additional stock
            quantity = predicted_demand * 0.3  # 30% of predicted demand
        else:
            # For other alert types, use a default quantity
            quantity = predicted_demand * 0.1

        # Round up to nearest whole number
        quantity = math.ceil(quantity)

        if quantity <= 0:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Warning'),
                    'message': _('Calculated order quantity is zero or negative'),
                    'sticky': False,
                    'type': 'warning',
                }
            }

        # Get the company from the warehouse
        company_id = self.warehouse_id.company_id.id or self.env.company.id

        # Find appropriate vendor
        seller = self.env['product.supplierinfo'].search([
            ('product_tmpl_id', '=', self.product_id.product_tmpl_id.id),
            ('company_id', '=', company_id)
        ], limit=1)

        # Vérifier si un fournisseur a été trouvé
        if not seller or not seller.partner_id:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Error'),
                    'message': _(
                        'No supplier found for this product. Please configure a supplier before creating a purchase order.'),
                    'sticky': False,
                    'type': 'danger',
                }
            }

        partner_id = seller.partner_id.id

        # Obtenir le type d'opération d'achat par défaut pour l'entrepôt
        picking_type = self.env['stock.picking.type'].search([
            ('code', '=', 'incoming'),
            ('warehouse_id', '=', self.warehouse_id.id),
            ('company_id', '=', company_id)
        ], limit=1)

        # Create purchase order
        purchase_order = self.env['purchase.order'].create({
            'partner_id': partner_id,
            'company_id': company_id,
            'picking_type_id': picking_type.id if picking_type else False,
            'origin': f'Alert #{self.id}: {self.name}',
            'date_order': fields.Datetime.now(),
            'user_id': self.env.user.id,
            'notes': f"Automatically generated from demand prediction alert.\n"
                     f"Alert type: {self.alert_type}\n"
                     f"Severity: {self.severity}\n"
                     f"Current stock: {current_stock}\n"
                     f"Predicted demand: {predicted_demand}\n"
                     f"Recommended action: {self.recommended_action}"
        })

        # Add product line
        self.env['purchase.order.line'].create({
            'order_id': purchase_order.id,
            'product_id': self.product_id.id,
            'name': self.product_id.name,
            'product_qty': quantity,
            'product_uom': self.product_id.uom_po_id.id or self.product_id.uom_id.id,
            'price_unit': seller.price if seller else self.product_id.standard_price,
            'date_planned': fields.Datetime.now(),
            'company_id': company_id,
        })

        # Log the action
        self.env['demand.prediction.alert.action'].create({
            'alert_id': self.id,
            'action_type': 'purchase_request',
            'description': _('Purchase order %s created with %s units') % (purchase_order.name, quantity),
            'user_id': self.env.user.id,
        })

        # Update alert state
        self.write({'state': 'acknowledged'})

        # Return action to view the created purchase order
        return {
            'type': 'ir.actions.act_window',
            'name': _('Purchase Order'),
            'res_model': 'purchase.order',
            'res_id': purchase_order.id,
            'view_mode': 'form',
            'view_type': 'form',
            'target': 'current',
        }

    def action_send_email_manually(self):
        """Action pour envoyer manuellement un email de notification"""
        self.ensure_one()

        # Use direct email sending
        if self._send_direct_email(self):
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Email Sent'),
                    'message': _('Email notification sent successfully to %s') % self.user_id.email,
                    'sticky': False,
                    'type': 'success',
                }
            }
        else:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Error'),
                    'message': _('Failed to send email notification'),
                    'sticky': False,
                    'type': 'danger',
                }
            }


class DemandPredictionAlertAction(models.Model):
    _name = 'demand.prediction.alert.action'
    _description = 'Demand Prediction Alert Action'
    _order = 'create_date desc'

    alert_id = fields.Many2one('demand.prediction.alert', string='Alert', required=True, ondelete='cascade')
    action_type = fields.Selection([
        ('notification', 'Notification'),
        ('status_change', 'Status Change'),
        ('purchase_request', 'Purchase Request'),
        ('note', 'Note'),
    ], string='Action Type', required=True)
    description = fields.Text('Description', required=True)
    user_id = fields.Many2one('res.users', string='User', required=True, default=lambda self: self.env.user)
    create_date = fields.Datetime('Date', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', related='alert_id.company_id', store=True)

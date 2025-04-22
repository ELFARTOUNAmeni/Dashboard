{
    'name': 'Sales Prediction',
    'version': '1.0',
    'category': 'Sales',
    'summary': 'Predict sales using XGBoost model',
    'description': """
        This module integrates a trained XGBoost model to predict sales for products.
        Features:
        - Sales forecasting for products
        - Inventory recommendations
        - Interactive dashboard
        - Model retraining capabilities
    """,
    'author': 'Elfartoun Ameni',
    'depends': ['base', 'sale', 'stock', 'product', 'web', 'mail',
                'sale_management',
                'stock',
                'web',
                ],
    'data': [
        'security/sales_prediction_security.xml',
        'security/ir.model.access.csv',
        'views/sales_prediction_views.xml',
        'wizard/retrain_model_views.xml',
        'wizard/generate_prediction_views.xml',
        'views/sales_dashboard_views.xml',
        'views/sales_prediction_dashboard_views.xml',
        'views/sales_prediction_summary_views.xml',
        'views/demand_prediction_views.xml',
        'wizard/demand_prediction_wizard_views.xml',
        'views/demand_prediction_actions.xml',
        'views/demand_prediction_menu.xml',
        'views/demand_prediction_alert_views.xml',
        'views/customer_segment_views.xml',
        'views/res_partner_views.xml',
        'views/product_recommendation_views.xml',
        'views/promotion_notification_views.xml',
        'views/product_promotion_views.xml',
        'views/customer_segmentation_dashboard_views.xml',
        'views/customer_segmentation_templates.xml',

        'wizard/generate_segments_views.xml',
        'wizard/get_recommendations_views.xml',
        'wizard/create_test_data_views.xml',
        'views/menu_views.xml',
        'data/ir_cron_data.xml',
        'data/cron_data_promo.xml',
        'data/mail_template_data.xml',
        'data/email_templates_promo.xml',


    ],'assets': {
    'web.assets_backend': [
        'sales_prediction/static/src/css/dashboard.css',
        'sales_prediction/static/src/js/dashboard.js',
        'sales_prediction/static/lib/chart/chart.js',
        '/sales_prediction/static/lib/chart/chart.min.js',
        'sales_prediction/static/src/xml/dashboard.xml',
        'sales_prediction/static/src/css/sales_dashboard.css',
        'sales_prediction/static/lib/chart/chart.js',
        'sales_prediction/static/lib/chart/chart.min.js',
        'sales_prediction/static/src/js/sales_dashboard.js',
        'sales_prediction/static/src/xml/sales_dashboard.xml',

    ],
},

    'installable': True,
    'application': True,
    'license': 'LGPL-3',
    'icon': '/sales_prediction/static/description/icon.png',
    'external_dependencies': {
        'python': ['sklearn', 'numpy', 'pandas'],
    },
}

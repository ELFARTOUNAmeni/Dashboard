from odoo import models, fields, api
import logging
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

_logger = logging.getLogger(__name__)


class GenerateSegmentsWizard(models.TransientModel):
    _name = 'generate.segments.wizard'
    _description = 'Generate Customer Segments'

    num_segments = fields.Integer(string='Number of Segments', default=3, required=True)
    min_orders = fields.Integer(string='Minimum Orders', default=1,
                                help='Minimum number of orders for a customer to be included')
    algorithm = fields.Selection([
        ('kmeans', 'K-Means'),
        ('gmm', 'Gaussian Mixture Model')
    ], string='Clustering Algorithm', default='kmeans', required=True,
        help='K-Means creates spherical clusters, GMM is more flexible with elliptical clusters')
    covariance_type = fields.Selection([
        ('full', 'Full'),
        ('tied', 'Tied'),
        ('diag', 'Diagonal'),
        ('spherical', 'Spherical')
    ], string='GMM Covariance Type', default='full',
        help='Controls the shape of the Gaussian distributions')

    def action_generate_segments(self):
        self.ensure_one()

        # Get customers with sufficient order history
        partners = self.env['res.partner'].search([
            ('customer_rank', '>', 0),
            ('order_count', '>=', self.min_orders)
        ])

        if len(partners) < self.num_segments:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Not Enough Data',
                    'message': f'Need at least {self.num_segments} customers with {self.min_orders} orders. Found only {len(partners)}. Use the "Create Test Data" menu to generate test data.',
                    'sticky': False,
                    'type': 'warning',
                }
            }

        try:
            # Prepare data for clustering
            features = []
            partner_ids = []

            for partner in partners:
                # Skip partners with missing or invalid data
                if partner.avg_order_value <= 0 or partner.order_frequency < 0:
                    continue

                features.append([
                    partner.avg_order_value,
                    partner.order_frequency,
                    partner.category_count,
                    partner.days_since_last_order,
                    partner.order_count
                ])
                partner_ids.append(partner.id)

            if len(features) < self.num_segments:
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Not Enough Valid Data',
                        'message': f'Need at least {self.num_segments} customers with valid purchase data. Found only {len(features)}.',
                        'sticky': False,
                        'type': 'warning',
                    }
                }

            X = np.array(features)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply selected clustering algorithm
            if self.algorithm == 'kmeans':
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=self.num_segments, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                cluster_centers_scaled = kmeans.cluster_centers_
                model = kmeans
            else:  # GMM
                # Apply Gaussian Mixture Model
                gmm = GaussianMixture(
                    n_components=self.num_segments,
                    covariance_type=self.covariance_type,
                    random_state=42,
                    n_init=10
                )
                gmm.fit(X_scaled)
                clusters = gmm.predict(X_scaled)
                # For GMM, use the means as cluster centers
                cluster_centers_scaled = gmm.means_
                model = gmm

            # Calculate and print accuracy metrics
            self._calculate_and_print_accuracy(X_scaled, clusters, model)

            # Create or update segments
            segments = self.env['customer.segment']
            existing_segments = self.env['customer.segment'].search([])

            # Archive old segments
            if existing_segments:
                existing_segments.write({'active': False})

            # Calculate global averages for comparison
            global_avg_order_value = np.mean([p.avg_order_value for p in partners if p.avg_order_value > 0])
            global_avg_order_frequency = np.mean([p.order_frequency for p in partners if p.order_frequency > 0])
            global_avg_days_since_last = np.mean(
                [p.days_since_last_order for p in partners if p.days_since_last_order >= 0])
            global_avg_order_count = np.mean([p.order_count for p in partners if p.order_count > 0])
            global_avg_category_count = np.mean([p.category_count for p in partners if p.category_count > 0])

            # Create new segments with meaningful names
            for i in range(self.num_segments):
                # Get centroid values (unstandardized)
                centroid_scaled = cluster_centers_scaled[i]
                centroid = scaler.inverse_transform([centroid_scaled])[0]

                # Assign meaningful name and description based on segment characteristics
                name, description = self._get_segment_name_and_description(
                    centroid[0],  # avg_order_value
                    centroid[1],  # order_frequency
                    centroid[2],  # category_count
                    centroid[3],  # days_since_last_order
                    centroid[4],  # order_count
                    global_avg_order_value,
                    global_avg_order_frequency,
                    global_avg_days_since_last,
                    global_avg_order_count,
                    global_avg_category_count
                )

                # Add algorithm info to description
                algorithm_info = "K-means clustering" if self.algorithm == 'kmeans' else f"Gaussian Mixture Model ({self.covariance_type} covariance)"
                description += f"\n\nGenerated using: {algorithm_info}"

                segment = self.env['customer.segment'].create({
                    'name': name,
                    'description': description,
                    'avg_order_value': centroid[0],
                    'avg_order_frequency': centroid[1],
                    'avg_product_categories': centroid[2],
                    'last_generated': fields.Datetime.now(),
                    'algorithm': self.algorithm,
                })
                segments += segment

            # Assign customers to segments and generate recommendations
            for i, partner_id in enumerate(partner_ids):
                cluster_id = clusters[i]
                self.env['res.partner'].browse(partner_id).write({'segment_id': segments[cluster_id].id})

            # Generate product recommendations for each segment
            self._generate_recommendations(segments)

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Success',
                    'message': f'Successfully generated {self.num_segments} customer segments using {self.algorithm.upper()}. Check server logs for accuracy metrics.',
                    'sticky': False,
                    'type': 'success',
                }
            }

        except Exception as e:
            _logger.error("Error generating segments: %s", str(e))
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Error',
                    'message': f'Error generating segments: {str(e)}',
                    'sticky': True,
                    'type': 'danger',
                }
            }

    def _calculate_and_print_accuracy(self, X, clusters, model):
        """Calculate and print various accuracy metrics for the clustering model"""
        print("\n" + "=" * 50)
        print(f"CLUSTERING ACCURACY METRICS - Algorithm: {self.algorithm.upper()}")
        print("=" * 50)

        # Metrics for both K-means and GMM
        try:
            # Silhouette score (range -1 to 1, higher is better)
            sil_score = silhouette_score(X, clusters)
            print(f"Silhouette Score: {sil_score:.4f} (higher is better, range -1 to 1)")
        except Exception as e:
            print(f"Error calculating silhouette score: {str(e)}")

        try:
            # Calinski-Harabasz Index (higher is better)
            ch_score = calinski_harabasz_score(X, clusters)
            print(f"Calinski-Harabasz Index: {ch_score:.2f} (higher is better)")
        except Exception as e:
            print(f"Error calculating Calinski-Harabasz score: {str(e)}")

        try:
            # Davies-Bouldin Index (lower is better)
            db_score = davies_bouldin_score(X, clusters)
            print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")
        except Exception as e:
            print(f"Error calculating Davies-Bouldin score: {str(e)}")

        # Algorithm-specific metrics
        if self.algorithm == 'kmeans':
            print("\nK-MEANS SPECIFIC METRICS:")
            print(f"Inertia: {model.inertia_:.2f} (lower is better)")

        if self.algorithm == 'gmm':
            print("\nGMM SPECIFIC METRICS:")
            try:
                print(f"Log-Likelihood: {model.score(X) * len(X):.2f} (higher is better)")
                print(f"BIC Score: {model.bic(X):.2f} (lower is better)")
                print(f"AIC Score: {model.aic(X):.2f} (lower is better)")

                # Print probabilities summary
                probs = model.predict_proba(X)
                avg_prob = np.mean(probs)
                max_prob = np.max(probs)
                min_prob = np.min(probs)
                print(f"Average Probability: {avg_prob:.4f}")
                print(f"Maximum Probability: {max_prob:.4f}")
                print(f"Minimum Probability: {min_prob:.4f}")
            except Exception as e:
                print(f"Error calculating GMM scores: {str(e)}")

        # Print cluster sizes
        unique, counts = np.unique(clusters, return_counts=True)
        print("\nCLUSTER SIZES:")
        for i, count in zip(unique, counts):
            print(f"Cluster {i}: {count} customers")

        print("=" * 50 + "\n")

    def _get_segment_name_and_description(self, avg_order_value, order_frequency, category_count,
                                          days_since_last, order_count, global_avg_order_value,
                                          global_avg_order_frequency, global_avg_days_since_last,
                                          global_avg_order_count, global_avg_category_count):
        """
        Generate meaningful segment name and description based on its characteristics
        compared to global averages
        """
        # Initialize score components
        value_score = 0  # High value vs low value
        frequency_score = 0  # Frequent vs occasional
        recency_score = 0  # Recent vs inactive
        diversity_score = 0  # Diverse vs focused
        loyalty_score = 0  # Loyal vs new

        # Calculate scores relative to global averages
        if avg_order_value > global_avg_order_value * 1.5:
            value_score = 2  # High value
        elif avg_order_value > global_avg_order_value * 0.8:
            value_score = 1  # Medium value
        else:
            value_score = -1  # Low value

        if order_frequency < global_avg_order_frequency * 0.5:
            frequency_score = 2  # Frequent buyer (lower days between orders)
        elif order_frequency < global_avg_order_frequency * 1.2:
            frequency_score = 1  # Regular buyer
        else:
            frequency_score = -1  # Occasional buyer

        if days_since_last < global_avg_days_since_last * 0.5:
            recency_score = 2  # Recent buyer
        elif days_since_last < global_avg_days_since_last * 1.2:
            recency_score = 1  # Moderately active
        else:
            recency_score = -1  # Inactive

        if category_count > global_avg_category_count * 1.5:
            diversity_score = 2  # Diverse buyer
        elif category_count > global_avg_category_count * 0.8:
            diversity_score = 1  # Moderately diverse
        else:
            diversity_score = -1  # Focused buyer

        if order_count > global_avg_order_count * 1.5:
            loyalty_score = 2  # Loyal customer
        elif order_count > global_avg_order_count * 0.8:
            loyalty_score = 1  # Regular customer
        else:
            loyalty_score = -1  # New customer

        # Determine primary and secondary characteristics
        scores = {
            'value': value_score,
            'frequency': frequency_score,
            'recency': recency_score,
            'diversity': diversity_score,
            'loyalty': loyalty_score
        }

        # Sort characteristics by score (highest first)
        sorted_chars = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_chars[0][0]
        secondary = sorted_chars[1][0]

        # Generate name based on primary and secondary characteristics
        name_components = {
            'value': {2: 'Premium', 1: 'Standard', -1: 'Budget'},
            'frequency': {2: 'Frequent', 1: 'Regular', -1: 'Occasional'},
            'recency': {2: 'Active', 1: 'Engaged', -1: 'Dormant'},
            'diversity': {2: 'Explorer', 1: 'Varied', -1: 'Focused'},
            'loyalty': {2: 'Loyal', 1: 'Returning', -1: 'New'}
        }

        primary_term = name_components[primary][scores[primary]]
        secondary_term = name_components[secondary][scores[secondary]]

        # Create segment name
        name = f"{primary_term} {secondary_term} Customers"

        # Create detailed description
        description = f"This segment consists of customers with the following characteristics:\n"

        # Add value description
        if value_score == 2:
            description += "- High average order value\n"
        elif value_score == 1:
            description += "- Medium average order value\n"
        else:
            description += "- Lower average order value\n"

        # Add frequency description
        if frequency_score == 2:
            description += "- Purchase frequently\n"
        elif frequency_score == 1:
            description += "- Purchase at regular intervals\n"
        else:
            description += "- Purchase occasionally\n"

        # Add recency description
        if recency_score == 2:
            description += "- Recently active\n"
        elif recency_score == 1:
            description += "- Moderately active\n"
        else:
            description += "- Not recently active\n"

        # Add diversity description
        if diversity_score == 2:
            description += "- Purchase from many product categories\n"
        elif diversity_score == 1:
            description += "- Purchase from several product categories\n"
        else:
            description += "- Focus on specific product categories\n"

        # Add loyalty description
        if loyalty_score == 2:
            description += "- Have made many purchases\n"
        elif loyalty_score == 1:
            description += "- Have made several purchases\n"
        else:
            description += "- Have made few purchases\n"

        # Add numerical details
        description += f"\nAverage order value: {avg_order_value:.2f}\n"
        description += f"Average days between orders: {order_frequency:.1f}\n"
        description += f"Average product categories: {category_count:.1f}\n"
        description += f"Average days since last order: {days_since_last:.1f}\n"
        description += f"Average number of orders: {order_count:.1f}"

        return name, description

    def _generate_recommendations(self, segments):
        """Generate product recommendations for each segment based on purchase history"""
        for segment in segments:
            # Clear existing recommendations
            segment.recommendation_ids.unlink()

            # Get all confirmed orders for customers in this segment
            partners = segment.partner_ids
            orders = self.env['sale.order'].search([
                ('partner_id', 'in', partners.ids),
                ('state', 'in', ['sale', 'done'])
            ])

            # Count product purchases
            product_counts = {}
            for order in orders:
                for line in order.order_line:
                    product_id = line.product_id.id
                    if product_id in product_counts:
                        product_counts[product_id] += line.product_uom_qty
                    else:
                        product_counts[product_id] = line.product_uom_qty

            # Create recommendations (top 10 products)
            recommendations = []
            for product_id, count in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                recommendations.append({
                    'segment_id': segment.id,
                    'product_id': product_id,
                    'score': float(count),
                    'purchase_count': int(count),
                })

            # Create recommendation records
            for rec in recommendations:
                self.env['product.recommendation'].create(rec)

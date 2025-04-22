from odoo import models, fields, api
import logging
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, precision_score, \
    recall_score, f1_score, ndcg_score

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
        """Generate pure ML-based product recommendations with segment-specific models"""
        from sklearn.decomposition import TruncatedSVD, NMF
        from sklearn.preprocessing import normalize
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        for segment in segments:
            # Clear existing recommendations
            segment.recommendation_ids.unlink()

            # Get all customers in this segment
            partners = segment.partner_ids
            if not partners:
                continue

            # Get all confirmed orders for customers in this segment
            orders = self.env['sale.order'].search([
                ('partner_id', 'in', partners.ids),
                ('state', 'in', ['sale', 'done'])
            ])

            if not orders:
                continue

            # Prepare data structures
            customers = partners.ids
            all_products = set()
            customer_products = {customer_id: {} for customer_id in customers}

            # Collect product metadata for content-based filtering
            product_categories = {}
            product_descriptions = {}

            for order in orders:
                customer_id = order.partner_id.id
                for line in order.order_line:
                    product_id = line.product_id.id
                    if product_id not in all_products:
                        all_products.add(product_id)
                        # Store product metadata
                        product = line.product_id
                        product_categories[product_id] = product.categ_id.id if product.categ_id else 0
                        product_descriptions[product_id] = f"{product.name} {product.description or ''}"

                    # Add or update quantity for this customer-product pair
                    if product_id in customer_products[customer_id]:
                        customer_products[customer_id][product_id] += line.product_uom_qty
                    else:
                        customer_products[customer_id][product_id] = line.product_uom_qty

            # Convert to list for indexing
            products_list = list(all_products)

            # Create interaction matrix
            matrix = np.zeros((len(customers), len(products_list)))
            for i, customer_id in enumerate(customers):
                for j, product_id in enumerate(products_list):
                    if product_id in customer_products[customer_id]:
                        matrix[i, j] = customer_products[customer_id][product_id]

            # Handle edge cases: not enough data
            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                # Fallback to a simple ML-based approach instead of frequency-based
                self._generate_fallback_ml_recommendations(segment, products_list, customer_products)
                continue

            try:
                # Choose algorithm based on segment characteristics
                algorithm = self._select_algorithm_for_segment(segment)

                final_scores = None
                model_object = None  # Store the model object

                if algorithm == 'svd':
                    # Collaborative filtering with SVD
                    n_components = min(min(matrix.shape) - 1, 10)
                    n_components = max(n_components, 2)  # Ensure at least 2 components

                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    latent_matrix = svd.fit_transform(matrix)

                    # Reconstruct the matrix
                    predicted_matrix = latent_matrix @ svd.components_

                    # Segment-wide product affinity
                    final_scores = np.sum(predicted_matrix, axis=0)
                    model_object = svd

                elif algorithm == 'nmf':
                    # Non-negative Matrix Factorization (better for sparse data)
                    n_components = min(min(matrix.shape) - 1, 8)
                    n_components = max(n_components, 2)

                    # Add small constant to avoid zeros
                    matrix_pos = matrix + 0.001

                    nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
                    latent_matrix = nmf.fit_transform(matrix_pos)

                    # Reconstruct the matrix
                    predicted_matrix = latent_matrix @ nmf.components_

                    # Segment-wide product affinity
                    final_scores = np.sum(predicted_matrix, axis=0)
                    model_object = nmf

                elif algorithm == 'content':
                    # Content-based filtering using product descriptions
                    if len(products_list) < 2:
                        self._generate_fallback_ml_recommendations(segment, products_list, customer_products)
                        continue

                    # Create document corpus from product descriptions
                    corpus = [product_descriptions.get(pid, '') for pid in products_list]

                    # TF-IDF vectorization
                    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
                    product_features = tfidf.fit_transform(corpus)

                    # Calculate similarity between products
                    product_similarity = product_features @ product_features.T

                    # Weighted sum based on current purchase patterns
                    purchase_weights = np.sum(matrix, axis=0)
                    normalized_weights = purchase_weights / np.sum(purchase_weights) if np.sum(
                        purchase_weights) > 0 else purchase_weights

                    # Calculate scores
                    final_scores = np.zeros(len(products_list))
                    for i, weight in enumerate(normalized_weights):
                        if weight > 0:
                            final_scores += weight * np.array(product_similarity[i].todense())[0]

                    model_object = tfidf

                elif algorithm == 'hybrid':
                    # Hybrid approach - combine collaborative and content-based
                    # Collaborative component
                    n_components = min(min(matrix.shape) - 1, 5)
                    n_components = max(n_components, 2)

                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    try:
                        latent_matrix = svd.fit_transform(matrix)
                        collab_scores = np.sum(latent_matrix @ svd.components_, axis=0)
                        model_object = svd
                    except:
                        # Use NMF as fallback for collaborative component
                        nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
                        matrix_pos = matrix + 0.001
                        try:
                            latent_matrix = nmf.fit_transform(matrix_pos)
                            collab_scores = np.sum(latent_matrix @ nmf.components_, axis=0)
                            model_object = nmf
                        except:
                            # If both fail, use matrix factorization on binary data
                            binary_matrix = (matrix > 0).astype(float)
                            svd_binary = TruncatedSVD(n_components=2, random_state=42)
                            latent_binary = svd_binary.fit_transform(binary_matrix)
                            collab_scores = np.sum(latent_binary @ svd_binary.components_, axis=0)
                            model_object = svd_binary

                    # Content component
                    corpus = [product_descriptions.get(pid, '') for pid in products_list]

                    # Handle empty descriptions
                    if all(not desc for desc in corpus):
                        # Use category information instead
                        content_scores = np.zeros(len(products_list))
                        for i, prod_id in enumerate(products_list):
                            cat_id = product_categories.get(prod_id, 0)
                            for j, other_prod_id in enumerate(products_list):
                                other_cat_id = product_categories.get(other_prod_id, 0)
                                # Simple category similarity
                                content_scores[j] += 1 if cat_id == other_cat_id and cat_id != 0 else 0
                    else:
                        try:
                            tfidf = TfidfVectorizer(max_features=50, stop_words='english')
                            product_features = tfidf.fit_transform(corpus)

                            # Use purchase history as query
                            purchase_weights = np.sum(matrix, axis=0)
                            content_scores = np.zeros(len(products_list))

                            # Simple product similarity
                            for i, prod_id in enumerate(products_list):
                                similarity = product_features @ product_features[i].T
                                content_scores += np.array(similarity.todense())[0]
                        except:
                            # Fallback to binary similarity
                            content_scores = np.ones(len(products_list))

                    # Normalize and combine scores
                    if np.sum(collab_scores) > 0:
                        collab_scores = collab_scores / np.sum(collab_scores)
                    if np.sum(content_scores) > 0:
                        content_scores = content_scores / np.sum(content_scores)

                    final_scores = 0.7 * collab_scores + 0.3 * content_scores

                elif algorithm == 'knn':
                    # K-Nearest Neighbors approach for sparse data
                    # Convert to binary matrix (purchased or not)
                    binary_matrix = (matrix > 0).astype(float)

                    # Use kNN for item similarity
                    n_neighbors = min(5, binary_matrix.shape[0] - 1)
                    n_neighbors = max(n_neighbors, 2)

                    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
                    knn.fit(binary_matrix.T)  # Fit on item features

                    # For each product, find similar products
                    distances, indices = knn.kneighbors(binary_matrix.T)

                    # Calculate weighted item similarity scores
                    purchase_counts = np.sum(binary_matrix, axis=0)
                    final_scores = np.zeros(len(products_list))

                    for i in range(len(products_list)):
                        for j, idx in enumerate(indices[i]):
                            if idx != i:  # Avoid self-similarity
                                # Weighted by purchase count and similarity (1 - distance)
                                similarity = 1 - distances[i, j]
                                final_scores[idx] += similarity * purchase_counts[i]

                    model_object = knn

                else:
                    # Default to a ML fallback
                    self._generate_fallback_ml_recommendations(segment, products_list, customer_products)
                    continue

                # Ensure we have scores
                if final_scores is None or len(final_scores) == 0:
                    self._generate_fallback_ml_recommendations(segment, products_list, customer_products)
                    continue

                # Get top product indices (up to 10)
                top_indices = np.argsort(-final_scores)[:10]

                # Create recommendation records
                recommendations = []
                for idx in top_indices:
                    product_id = products_list[idx]
                    score = float(final_scores[idx])
                    actual_purchases = float(np.sum(matrix[:, idx]))

                    recommendations.append({
                        'segment_id': segment.id,
                        'product_id': product_id,
                        'score': score,
                        'purchase_count': int(actual_purchases),
                        'algorithm': algorithm  # Store the algorithm used
                    })

                # Create recommendation records
                for rec in recommendations:
                    self.env['product.recommendation'].create(rec)

                # Only call evaluate_model_performance if we have both matrix and final_scores
                if model_object is not None and final_scores is not None:
                    try:
                        # Create proper test data format
                        y_test = matrix  # Use actual purchase data as test
                        # Make predictions to evaluate
                        y_pred = np.zeros_like(matrix)
                        for i in range(matrix.shape[0]):
                            y_pred[i] = final_scores  # Use final scores as predictions

                        self.evaluate_model_performance(model_object, y_test, y_pred, segment.name, algorithm)
                    except Exception as e:
                        _logger.error(f"Error evaluating model performance: {str(e)}")
                        # Continue even if evaluation fails


            except Exception as e:
                _logger.error(f"Error generating recommendations for segment {segment.name}: {str(e)}")
                # Fallback to ML-based recommendations
                self._generate_fallback_ml_recommendations(segment, products_list, customer_products)

    def evaluate_model_performance(self, model, y_test, y_pred, segment_name, algorithm_name):
        """
        Evaluates and displays performance metrics for the recommendation model
        """
        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score

        print("\n" + "=" * 60)
        print(f"PERFORMANCE METRICS - Segment: {segment_name} - Algorithm: {algorithm_name}")
        print("=" * 60)

        # Metrics for all recommendation models
        try:
            # Precision and recall at binary level (predicted purchase vs actual)
            y_binary_true = (y_test > 0).astype(int)
            y_binary_pred = (y_pred > 0).astype(int)

            precision = precision_score(y_binary_true.flatten(), y_binary_pred.flatten(), zero_division=0)
            recall = recall_score(y_binary_true.flatten(), y_binary_pred.flatten(), zero_division=0)
            f1 = f1_score(y_binary_true.flatten(), y_binary_pred.flatten(), zero_division=0)

            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
        except Exception as e:
            print(f"Error calculating classification metrics: {str(e)}")
            precision, recall, f1 = 0, 0, 0

        # Metrics specific to ranking models
        try:
            # NDCG (Normalized Discounted Cumulative Gain)
            if hasattr(y_test, 'shape') and len(y_test.shape) > 1:
                k_values = [min(5, y_test.shape[1]), min(10, y_test.shape[1])]
                for k in k_values:
                    if k > 0:  # Ensure k is positive
                        try:
                            ndcg = ndcg_score(y_test, y_pred, k=k)
                            print(f"NDCG@{k}: {ndcg:.4f}")
                        except Exception as e:
                            print(f"Error calculating NDCG@{k}: {str(e)}")
        except Exception as e:
            print(f"Error calculating NDCG: {str(e)}")

        # Coverage and diversity of recommendations
        try:
            # Coverage: percentage of recommended items among all available items
            all_items = self.env['product.product'].search_count([])
            recommended_items = len(np.unique(np.where(y_pred > 0)[1])) if hasattr(y_pred, 'shape') and len(
                y_pred.shape) > 1 else 0
            coverage = recommended_items / all_items if all_items > 0 else 0
            print(f"Catalog coverage: {coverage:.2%}")
        except Exception as e:
            print(f"Error calculating coverage and diversity: {str(e)}")
            coverage = 0

        # Algorithm-specific metrics
        try:
            if algorithm_name == 'svd':
                print("\nSVD SPECIFIC METRICS:")
                # Explained variance
                if hasattr(model, 'explained_variance_ratio_'):
                    variance = sum(model.explained_variance_ratio_)
                    print(f"Explained variance: {variance:.2%}")

            elif algorithm_name in ['gmm', 'nmf']:
                print(f"\n{algorithm_name.upper()} SPECIFIC METRICS:")
                reconstruction_error = np.mean((y_test - y_pred) ** 2)
                print(f"Reconstruction error MSE: {reconstruction_error:.4f}")
        except Exception as e:
            print(f"Error calculating algorithm-specific metrics: {str(e)}")

        # Operational performance
        print("\nOPERATIONAL PERFORMANCE:")
        import time
        try:
            # Simple timing test for the model
            start_time = time.time()
            # Generic operation appropriate for most models:
            if hasattr(model, 'predict'):
                if hasattr(y_test, 'shape') and len(y_test.shape) > 1:
                    model.predict(y_test[:1]) if y_test.shape[0] > 0 else None
            elif hasattr(model, 'transform'):
                if hasattr(y_test, 'shape') and len(y_test.shape) > 1:
                    model.transform(y_test[:1]) if y_test.shape[0] > 0 else None
            inference_time = time.time() - start_time
            print(f"Average recommendation generation time: {inference_time:.4f} seconds per user")
        except Exception as e:
            print(f"Error measuring inference time: {str(e)}")
            inference_time = 0

        print("=" * 60 + "\n")

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coverage': coverage,
            'inference_time': inference_time
        }
    def _generate_fallback_ml_recommendations(self, segment, products_list, customer_products):
        """Fallback ML-based recommendation when primary methods fail"""
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler

        try:
            # Use a simple item popularity approach with ML normalization
            popularity_scores = {}
            category_boost = {}
            recency_scores = {}

            # Get product data
            products = self.env['product.product'].browse(products_list)
            for product in products:
                # Get sales history
                sale_lines = self.env['sale.order.line'].search([
                    ('product_id', '=', product.id),
                    ('order_id.state', 'in', ['sale', 'done'])
                ])

                # Calculate popularity based on:
                # 1. Total quantity sold
                qty_sold = sum(line.product_uom_qty for line in sale_lines)

                # 2. Category popularity boost
                categ_id = product.categ_id.id if product.categ_id else 0
                if categ_id in category_boost:
                    category_boost[categ_id] += qty_sold
                else:
                    category_boost[categ_id] = qty_sold

                # 3. Recency of purchases (more recent = higher score)
                recent_orders = sorted([
                    (line.order_id.date_order or line.order_id.create_date)
                    for line in sale_lines
                ], reverse=True)

                # Calculate days since most recent purchase
                if recent_orders:
                    from datetime import datetime
                    most_recent = recent_orders[0]
                    days_since = (datetime.now() - most_recent).days
                    recency_score = max(1, 100 - days_since) / 100  # Normalize to 0-1
                else:
                    recency_score = 0

                recency_scores[product.id] = recency_score
                popularity_scores[product.id] = qty_sold

            # Normalize popularity scores using ML scaling
            if popularity_scores:
                # Convert to array for scaling
                product_ids = list(popularity_scores.keys())
                popularity_values = np.array([popularity_scores[pid] for pid in product_ids]).reshape(-1, 1)

                # Apply Min-Max scaling
                scaler = MinMaxScaler()
                normalized_values = scaler.fit_transform(popularity_values).flatten()

                # Update with normalized values
                for i, pid in enumerate(product_ids):
                    popularity_scores[pid] = normalized_values[i]

            # Apply category boost
            for product_id in products_list:
                product = self.env['product.product'].browse(product_id)
                categ_id = product.categ_id.id if product.categ_id else 0

                # Get total category popularity
                categ_popularity = category_boost.get(categ_id, 0)

                # Normalize category popularity (0-0.5 range)
                if sum(category_boost.values()) > 0:
                    normalized_categ = 0.5 * categ_popularity / max(category_boost.values())
                else:
                    normalized_categ = 0

                # Calculate final score with category boost and recency
                base_score = popularity_scores.get(product_id, 0)
                recency = recency_scores.get(product_id, 0)

                # Weighted combination (ML-style ensemble)
                final_score = (0.5 * base_score) + (0.3 * normalized_categ) + (0.2 * recency)

                popularity_scores[product_id] = final_score

            # Create recommendations (top 10 products)
            recommendations = []
            for product_id, score in sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
                # Get actual purchase count
                purchase_count = 0
                for customer_purchases in customer_products.values():
                    if product_id in customer_purchases:
                        purchase_count += customer_purchases[product_id]

                recommendations.append({
                    'segment_id': segment.id,
                    'product_id': product_id,
                    'score': float(score),
                    'purchase_count': int(purchase_count),
                    'algorithm': 'ml_popularity'  # ML-enhanced popularity model
                })

            # Create recommendation records
            for rec in recommendations:
                self.env['product.recommendation'].create(rec)

        except Exception as e:
            _logger.error(f"Error in fallback ML recommendations for segment {segment.name}: {str(e)}")
            # If all else fails, create empty recommendations
            pass

    def _select_algorithm_for_segment(self, segment):
        """Select the best ML algorithm based on segment characteristics"""
        # High-value customers with diverse purchases: SVD works well
        if segment.avg_order_value > 100 and segment.avg_product_categories > 3:
            return 'svd'

        # Focused buyers with specific interests: Content-based works well
        elif segment.avg_product_categories < 2:
            return 'content'

        # New customers with few purchases: Hybrid approach
        elif segment.customer_count < 10 or segment.avg_order_frequency > 60:
            return 'hybrid'

        # Small segments with sparse data: KNN works well
        elif segment.customer_count < 30:
            return 'knn'

        # Customers with regular, moderate purchasing: NMF works well
        else:
            return 'nmf'


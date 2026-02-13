"""
Technical Support Ticket Classification and Knowledge Base Recommendation System

This module implements an intelligent ticket routing and knowledge base recommendation
system using domain-adapted language models and semantic similarity matching.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# NLP and Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TicketData:
    """Data class for support ticket information"""
    ticket_id: str
    title: str
    description: str
    category: str
    priority: str
    resolution_time: float  # in hours
    resolution_notes: Optional[str] = None
    
    @property
    def full_text(self) -> str:
        """Combine title and description for processing"""
        return f"{self.title}. {self.description}"


@dataclass
class KnowledgeBaseArticle:
    """Data class for knowledge base articles"""
    article_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    resolution_count: int = 0  # How many tickets this has helped resolve
    
    @property
    def full_text(self) -> str:
        """Combine title and content for embedding"""
        tags_text = " ".join(self.tags)
        return f"{self.title}. {self.content}. Tags: {tags_text}"


class EmbeddingGenerator:
    """
    Generates embeddings using domain-adapted language models.
    Supports multiple embedding strategies for technical content.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        use_tfidf: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace sentence-transformers model name
            use_tfidf: Whether to also generate TF-IDF features
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.use_tfidf = use_tfidf
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        if use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
        else:
            self.tfidf_vectorizer = None
            
        self.is_fitted = False
        
    def fit_tfidf(self, texts: List[str]) -> 'EmbeddingGenerator':
        """Fit TF-IDF vectorizer on training texts"""
        if self.tfidf_vectorizer is not None:
            logger.info("Fitting TF-IDF vectorizer...")
            self.tfidf_vectorizer.fit(texts)
            self.is_fitted = True
        return self
    
    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate sentence embeddings for texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.sentence_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def generate_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF features"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not enabled")
        
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        
        logger.info(f"Generating TF-IDF features for {len(texts)} texts...")
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def generate_combined_features(
        self, 
        texts: List[str],
        tfidf_weight: float = 0.3
    ) -> np.ndarray:
        """
        Generate combined embeddings and TF-IDF features
        
        Args:
            texts: List of text strings
            tfidf_weight: Weight for TF-IDF features (0-1)
            
        Returns:
            Combined feature array
        """
        embeddings = self.generate_embeddings(texts)
        
        if self.use_tfidf and self.is_fitted:
            tfidf_features = self.generate_tfidf_features(texts)
            
            # Normalize both feature sets
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
            tfidf_features = tfidf_features / (np.linalg.norm(tfidf_features, axis=1, keepdims=True) + 1e-10)
            
            # Combine with weighting
            combined = np.hstack([
                embeddings * (1 - tfidf_weight),
                tfidf_features * tfidf_weight
            ])
            return combined
        
        return embeddings


class TicketClassifier:
    """
    Multi-class ticket classification system with support for
    multiple classification algorithms and ensemble methods.
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        classifier_type: str = 'gradient_boosting'
    ):
        """
        Initialize ticket classifier
        
        Args:
            embedding_generator: Pre-configured embedding generator
            classifier_type: Type of classifier ('random_forest', 'gradient_boosting', 
                           'logistic_regression', 'ensemble')
        """
        self.embedding_generator = embedding_generator
        self.classifier_type = classifier_type
        self.label_encoder = LabelEncoder()
        
        # Initialize classifier based on type
        self.classifier = self._create_classifier(classifier_type)
        self.is_trained = False
        
        # Store training metadata
        self.training_metadata = {}
        
    def _create_classifier(self, classifier_type: str):
        """Create classifier instance based on type"""
        classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
        }
        
        if classifier_type not in classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        logger.info(f"Created {classifier_type} classifier")
        return classifiers[classifier_type]
    
    def train(
        self,
        tickets: List[TicketData],
        validation_split: float = 0.2,
        use_cross_validation: bool = True
    ) -> Dict[str, float]:
        """
        Train the ticket classification model
        
        Args:
            tickets: List of training tickets
            validation_split: Fraction of data for validation
            use_cross_validation: Whether to perform cross-validation
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training classifier on {len(tickets)} tickets...")
        
        # Extract texts and labels
        texts = [ticket.full_text for ticket in tickets]
        labels = [ticket.category for ticket in tickets]
        
        # Fit TF-IDF if needed
        if self.embedding_generator.use_tfidf:
            self.embedding_generator.fit_tfidf(texts)
        
        # Generate features
        X = self.embedding_generator.generate_combined_features(texts)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Train classifier
        logger.info("Training classifier...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        y_pred = self.classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred)
        
        metrics = {
            'validation_accuracy': val_accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'num_categories': len(self.label_encoder.classes_)
        }
        
        # Cross-validation
        if use_cross_validation:
            logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(
                self.classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
            )
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Detailed classification report
        y_val_pred = self.classifier.predict(X_val)
        report = classification_report(
            y_val, y_val_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        self.training_metadata = {
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            'classification_report': report,
            'categories': self.label_encoder.classes_.tolist()
        }
        
        logger.info(f"Training completed. Validation accuracy: {val_accuracy:.4f}")
        return metrics
    
    def predict(
        self,
        tickets: Union[TicketData, List[TicketData]],
        return_probabilities: bool = False
    ) -> Union[str, List[str], Tuple[List[str], np.ndarray]]:
        """
        Predict categories for tickets
        
        Args:
            tickets: Single ticket or list of tickets
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Predicted categories (and probabilities if requested)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Handle single ticket
        single_ticket = not isinstance(tickets, list)
        if single_ticket:
            tickets = [tickets]
        
        # Generate features
        texts = [ticket.full_text for ticket in tickets]
        X = self.embedding_generator.generate_combined_features(texts)
        
        # Predict
        y_pred = self.classifier.predict(X)
        predictions = self.label_encoder.inverse_transform(y_pred)
        
        if return_probabilities:
            probabilities = self.classifier.predict_proba(X)
            if single_ticket:
                return predictions[0], probabilities[0]
            return predictions.tolist(), probabilities
        
        return predictions[0] if single_ticket else predictions.tolist()
    
    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Get feature importance for tree-based classifiers"""
        if not hasattr(self.classifier, 'feature_importances_'):
            logger.warning("Classifier does not support feature importance")
            return None
        
        importances = self.classifier.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return pd.DataFrame({
            'feature_index': indices,
            'importance': importances[indices]
        })


class KnowledgeBaseRecommender:
    """
    Semantic knowledge base article recommendation system using
    vector similarity and relevance ranking.
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        similarity_metric: str = 'cosine'
    ):
        """
        Initialize knowledge base recommender
        
        Args:
            embedding_generator: Pre-configured embedding generator
            similarity_metric: Similarity metric ('cosine', 'euclidean')
        """
        self.embedding_generator = embedding_generator
        self.similarity_metric = similarity_metric
        
        self.articles: List[KnowledgeBaseArticle] = []
        self.article_embeddings: Optional[np.ndarray] = None
        self.article_index: Dict[str, int] = {}
        
        self.is_indexed = False
        
    def index_articles(self, articles: List[KnowledgeBaseArticle]) -> None:
        """
        Index knowledge base articles for fast retrieval
        
        Args:
            articles: List of knowledge base articles
        """
        logger.info(f"Indexing {len(articles)} knowledge base articles...")
        
        self.articles = articles
        self.article_index = {article.article_id: idx for idx, article in enumerate(articles)}
        
        # Generate embeddings for all articles
        texts = [article.full_text for article in articles]
        self.article_embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Normalize for cosine similarity
        if self.similarity_metric == 'cosine':
            norms = np.linalg.norm(self.article_embeddings, axis=1, keepdims=True)
            self.article_embeddings = self.article_embeddings / (norms + 1e-10)
        
        self.is_indexed = True
        logger.info("Indexing completed")
    
    def recommend(
        self,
        ticket: TicketData,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Tuple[KnowledgeBaseArticle, float]]:
        """
        Recommend knowledge base articles for a ticket
        
        Args:
            ticket: Support ticket
            top_k: Number of recommendations to return
            category_filter: Only recommend articles from this category
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (article, similarity_score) tuples
        """
        if not self.is_indexed:
            raise ValueError("Articles not indexed. Call index_articles() first.")
        
        # Generate ticket embedding
        ticket_embedding = self.embedding_generator.generate_embeddings([ticket.full_text])[0]
        
        # Normalize for cosine similarity
        if self.similarity_metric == 'cosine':
            ticket_embedding = ticket_embedding / (np.linalg.norm(ticket_embedding) + 1e-10)
        
        # Compute similarities
        if self.similarity_metric == 'cosine':
            similarities = np.dot(self.article_embeddings, ticket_embedding)
        elif self.similarity_metric == 'euclidean':
            similarities = -np.linalg.norm(
                self.article_embeddings - ticket_embedding, axis=1
            )
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Apply category filter if specified
        if category_filter:
            mask = np.array([
                article.category == category_filter 
                for article in self.articles
            ])
            similarities = np.where(mask, similarities, -np.inf)
        
        # Apply minimum similarity threshold
        similarities = np.where(similarities >= min_similarity, similarities, -np.inf)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter out invalid similarities
        recommendations = [
            (self.articles[idx], similarities[idx])
            for idx in top_indices
            if similarities[idx] > -np.inf
        ]
        
        return recommendations
    
    def batch_recommend(
        self,
        tickets: List[TicketData],
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[List[Tuple[KnowledgeBaseArticle, float]]]:
        """
        Recommend articles for multiple tickets efficiently
        
        Args:
            tickets: List of support tickets
            top_k: Number of recommendations per ticket
            category_filter: Optional category filter
            
        Returns:
            List of recommendation lists
        """
        if not self.is_indexed:
            raise ValueError("Articles not indexed. Call index_articles() first.")
        
        # Generate embeddings for all tickets
        texts = [ticket.full_text for ticket in tickets]
        ticket_embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Normalize for cosine similarity
        if self.similarity_metric == 'cosine':
            norms = np.linalg.norm(ticket_embeddings, axis=1, keepdims=True)
            ticket_embeddings = ticket_embeddings / (norms + 1e-10)
        
        # Compute similarity matrix (tickets x articles)
        if self.similarity_metric == 'cosine':
            similarity_matrix = np.dot(ticket_embeddings, self.article_embeddings.T)
        else:
            # Euclidean distance
            similarity_matrix = -np.sqrt(
                np.sum(ticket_embeddings[:, :, np.newaxis]**2, axis=1) +
                np.sum(self.article_embeddings.T**2, axis=0) -
                2 * np.dot(ticket_embeddings, self.article_embeddings.T)
            )
        
        # Apply category filter if specified
        if category_filter:
            mask = np.array([
                article.category == category_filter 
                for article in self.articles
            ])
            similarity_matrix = np.where(mask, similarity_matrix, -np.inf)
        
        # Get top-k for each ticket
        all_recommendations = []
        for i, ticket in enumerate(tickets):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            recommendations = [
                (self.articles[idx], similarities[idx])
                for idx in top_indices
                if similarities[idx] > -np.inf
            ]
            all_recommendations.append(recommendations)
        
        return all_recommendations


class PerformanceEvaluator:
    """
    Comprehensive evaluation of the support ticket system including
    classification accuracy, recommendation relevance, and resolution time impact.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_classifier(
        self,
        classifier: TicketClassifier,
        test_tickets: List[TicketData]
    ) -> Dict[str, any]:
        """
        Evaluate classifier performance
        
        Args:
            classifier: Trained ticket classifier
            test_tickets: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating classifier on {len(test_tickets)} test tickets...")
        
        # Get predictions
        texts = [ticket.full_text for ticket in test_tickets]
        true_labels = [ticket.category for ticket in test_tickets]
        
        predictions, probabilities = classifier.predict(test_tickets, return_probabilities=True)
        
        # Encode labels
        true_encoded = classifier.label_encoder.transform(true_labels)
        pred_encoded = classifier.label_encoder.transform(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(true_encoded, pred_encoded)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_encoded, pred_encoded, average='weighted', zero_division=0
        )
        
        # Per-category metrics
        report = classification_report(
            true_encoded, pred_encoded,
            target_names=classifier.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_encoded, pred_encoded)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'test_samples': len(test_tickets),
            'categories': classifier.label_encoder.classes_.tolist()
        }
        
        logger.info(f"Classifier Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1-Score: {f1:.4f}")
        
        return metrics
    
    def evaluate_recommender(
        self,
        recommender: KnowledgeBaseRecommender,
        test_tickets: List[TicketData],
        ground_truth_articles: Dict[str, List[str]],  # ticket_id -> list of relevant article_ids
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate recommender performance using ranking metrics
        
        Args:
            recommender: Trained recommender
            test_tickets: Test tickets
            ground_truth_articles: Mapping of ticket IDs to relevant article IDs
            top_k: Number of recommendations to evaluate
            
        Returns:
            Dictionary of recommendation metrics
        """
        logger.info(f"Evaluating recommender on {len(test_tickets)} test tickets...")
        
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        mrr_scores = []
        
        for ticket in test_tickets:
            if ticket.ticket_id not in ground_truth_articles:
                continue
            
            relevant_ids = set(ground_truth_articles[ticket.ticket_id])
            if not relevant_ids:
                continue
            
            # Get recommendations
            recommendations = recommender.recommend(ticket, top_k=top_k)
            recommended_ids = [article.article_id for article, _ in recommendations]
            
            # Calculate precision@k
            hits = len(set(recommended_ids) & relevant_ids)
            precision_at_k.append(hits / min(len(recommended_ids), top_k))
            
            # Calculate recall@k
            recall_at_k.append(hits / len(relevant_ids))
            
            # Calculate NDCG@k
            dcg = sum([
                1.0 / np.log2(i + 2) if rec_id in relevant_ids else 0.0
                for i, rec_id in enumerate(recommended_ids[:top_k])
            ])
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), top_k))])
            ndcg_at_k.append(dcg / idcg if idcg > 0 else 0.0)
            
            # Calculate MRR (Mean Reciprocal Rank)
            for i, rec_id in enumerate(recommended_ids):
                if rec_id in relevant_ids:
                    mrr_scores.append(1.0 / (i + 1))
                    break
            else:
                mrr_scores.append(0.0)
        
        metrics = {
            f'precision@{top_k}': np.mean(precision_at_k) if precision_at_k else 0.0,
            f'recall@{top_k}': np.mean(recall_at_k) if recall_at_k else 0.0,
            f'ndcg@{top_k}': np.mean(ndcg_at_k) if ndcg_at_k else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'evaluated_tickets': len(precision_at_k)
        }
        
        logger.info(f"Precision@{top_k}: {metrics[f'precision@{top_k}']:.4f}")
        logger.info(f"NDCG@{top_k}: {metrics[f'ndcg@{top_k}']:.4f}")
        logger.info(f"MRR: {metrics['mrr']:.4f}")
        
        return metrics
    
    def evaluate_resolution_time_impact(
        self,
        baseline_tickets: List[TicketData],
        improved_tickets: List[TicketData]
    ) -> Dict[str, float]:
        """
        Evaluate impact on resolution time
        
        Args:
            baseline_tickets: Tickets without system assistance
            improved_tickets: Tickets with system assistance
            
        Returns:
            Dictionary of resolution time metrics
        """
        baseline_times = [t.resolution_time for t in baseline_tickets]
        improved_times = [t.resolution_time for t in improved_tickets]
        
        baseline_mean = np.mean(baseline_times)
        improved_mean = np.mean(improved_times)
        
        reduction_percentage = ((baseline_mean - improved_mean) / baseline_mean) * 100
        
        metrics = {
            'baseline_mean_hours': baseline_mean,
            'improved_mean_hours': improved_mean,
            'reduction_hours': baseline_mean - improved_mean,
            'reduction_percentage': reduction_percentage,
            'baseline_median_hours': np.median(baseline_times),
            'improved_median_hours': np.median(improved_times)
        }
        
        logger.info(f"Average resolution time reduced by {reduction_percentage:.2f}%")
        logger.info(f"From {baseline_mean:.2f}h to {improved_mean:.2f}h")
        
        return metrics
    
    def generate_evaluation_report(
        self,
        classifier_metrics: Dict,
        recommender_metrics: Dict,
        resolution_metrics: Optional[Dict] = None
    ) -> Dict:
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'classification': classifier_metrics,
            'recommendation': recommender_metrics
        }
        
        if resolution_metrics:
            report['resolution_time'] = resolution_metrics
        
        return report


class TechnicalSupportSystem:
    """
    Main system class that integrates classification and recommendation
    for production deployment.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        classifier_type: str = 'gradient_boosting',
        use_tfidf: bool = True
    ):
        """
        Initialize the technical support system
        
        Args:
            embedding_model: Sentence transformer model name
            classifier_type: Type of classifier to use
            use_tfidf: Whether to use TF-IDF features
        """
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            use_tfidf=use_tfidf
        )
        
        self.classifier = TicketClassifier(
            self.embedding_generator,
            classifier_type=classifier_type
        )
        
        self.recommender = KnowledgeBaseRecommender(
            self.embedding_generator,
            similarity_metric='cosine'
        )
        
        self.evaluator = PerformanceEvaluator()
        
        logger.info("Technical Support System initialized")
    
    def train(
        self,
        training_tickets: List[TicketData],
        knowledge_base: List[KnowledgeBaseArticle],
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the complete system
        
        Args:
            training_tickets: Training ticket dataset
            knowledge_base: Knowledge base articles
            validation_split: Validation data fraction
            
        Returns:
            Training metrics
        """
        logger.info("Training Technical Support System...")
        
        # Train classifier
        classifier_metrics = self.classifier.train(
            training_tickets,
            validation_split=validation_split,
            use_cross_validation=True
        )
        
        # Index knowledge base
        self.recommender.index_articles(knowledge_base)
        
        training_report = {
            'timestamp': datetime.now().isoformat(),
            'classifier_metrics': classifier_metrics,
            'knowledge_base_size': len(knowledge_base),
            'training_tickets': len(training_tickets)
        }
        
        logger.info("Training completed successfully")
        return training_report
    
    def process_ticket(
        self,
        ticket: TicketData,
        recommend_articles: bool = True,
        top_k: int = 5
    ) -> Dict:
        """
        Process a support ticket: classify and recommend articles
        
        Args:
            ticket: Support ticket to process
            recommend_articles: Whether to recommend KB articles
            top_k: Number of article recommendations
            
        Returns:
            Processing results
        """
        # Classify ticket
        predicted_category, probabilities = self.classifier.predict(
            ticket,
            return_probabilities=True
        )
        
        # Get top 3 category predictions with confidence
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                'category': self.classifier.label_encoder.classes_[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        result = {
            'ticket_id': ticket.ticket_id,
            'predicted_category': predicted_category,
            'confidence': float(probabilities[np.argmax(probabilities)]),
            'top_predictions': top_predictions,
            'recommendations': []
        }
        
        # Recommend articles
        if recommend_articles:
            recommendations = self.recommender.recommend(
                ticket,
                top_k=top_k,
                category_filter=predicted_category
            )
            
            result['recommendations'] = [
                {
                    'article_id': article.article_id,
                    'title': article.title,
                    'category': article.category,
                    'similarity_score': float(score),
                    'tags': article.tags
                }
                for article, score in recommendations
            ]
        
        return result
    
    def save(self, save_dir: str) -> None:
        """
        Save the trained system to disk
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving system to {save_dir}...")
        
        # Save classifier
        with open(save_path / 'classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Save recommender
        with open(save_path / 'recommender.pkl', 'wb') as f:
            pickle.dump(self.recommender, f)
        
        # Save metadata
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'embedding_model': self.embedding_generator.model_name,
            'classifier_type': self.classifier.classifier_type,
            'training_metadata': self.classifier.training_metadata
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("System saved successfully")
    
    @classmethod
    def load(cls, load_dir: str) -> 'TechnicalSupportSystem':
        """
        Load a trained system from disk
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            Loaded TechnicalSupportSystem
        """
        load_path = Path(load_dir)
        logger.info(f"Loading system from {load_dir}...")
        
        # Load metadata
        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        system = cls(
            embedding_model=metadata['embedding_model'],
            classifier_type=metadata['classifier_type']
        )
        
        # Load classifier
        with open(load_path / 'classifier.pkl', 'rb') as f:
            system.classifier = pickle.load(f)
        
        # Load recommender
        with open(load_path / 'recommender.pkl', 'rb') as f:
            system.recommender = pickle.load(f)
        
        logger.info("System loaded successfully")
        return system

"""
Unit Tests for Technical Support Classification System

Run with: python -m pytest test_system.py -v
"""

import pytest
import numpy as np
from typing import List

from ticket_classifier import (
    TicketData,
    KnowledgeBaseArticle,
    EmbeddingGenerator,
    TicketClassifier,
    KnowledgeBaseRecommender,
    PerformanceEvaluator,
    TechnicalSupportSystem
)


# Test Fixtures
@pytest.fixture
def sample_tickets() -> List[TicketData]:
    """Generate sample tickets for testing"""
    return [
        TicketData(
            ticket_id="T-001",
            title="VPN connection timeout",
            description="Cannot connect to VPN. Getting timeout errors repeatedly.",
            category="Network",
            priority="High",
            resolution_time=3.5
        ),
        TicketData(
            ticket_id="T-002",
            title="Laptop screen flickering",
            description="Monitor display shows flickering. Hardware issue suspected.",
            category="Hardware",
            priority="Medium",
            resolution_time=8.0
        ),
        TicketData(
            ticket_id="T-003",
            title="Software installation failed",
            description="Application install crashes. Error code 500.",
            category="Software",
            priority="Low",
            resolution_time=2.5
        ),
        TicketData(
            ticket_id="T-004",
            title="Database query slow",
            description="Production queries timing out. Performance degradation.",
            category="Database",
            priority="High",
            resolution_time=6.0
        ),
        TicketData(
            ticket_id="T-005",
            title="Access denied error",
            description="Cannot access shared folder. Permission denied.",
            category="Security",
            priority="Medium",
            resolution_time=4.5
        ),
    ]


@pytest.fixture
def sample_kb_articles() -> List[KnowledgeBaseArticle]:
    """Generate sample knowledge base articles"""
    return [
        KnowledgeBaseArticle(
            article_id="KB-001",
            title="VPN Troubleshooting Guide",
            content="Steps to resolve VPN connectivity issues including timeout errors.",
            category="Network",
            tags=["vpn", "connection", "troubleshoot"],
            resolution_count=25
        ),
        KnowledgeBaseArticle(
            article_id="KB-002",
            title="Monitor Display Issues",
            content="How to fix monitor flickering and display problems.",
            category="Hardware",
            tags=["monitor", "display", "hardware"],
            resolution_count=15
        ),
        KnowledgeBaseArticle(
            article_id="KB-003",
            title="Software Installation Problems",
            content="Resolving common software installation errors and failures.",
            category="Software",
            tags=["install", "software", "error"],
            resolution_count=30
        ),
        KnowledgeBaseArticle(
            article_id="KB-004",
            title="Database Performance Tuning",
            content="Optimizing database query performance and reducing timeouts.",
            category="Database",
            tags=["database", "performance", "query"],
            resolution_count=20
        ),
        KnowledgeBaseArticle(
            article_id="KB-005",
            title="Access Control Management",
            content="Managing user permissions and resolving access denied errors.",
            category="Security",
            tags=["access", "permissions", "security"],
            resolution_count=18
        ),
    ]


@pytest.fixture
def embedding_generator():
    """Create embedding generator"""
    return EmbeddingGenerator(
        model_name='all-MiniLM-L6-v2',
        use_tfidf=True
    )


# Tests for EmbeddingGenerator
class TestEmbeddingGenerator:
    
    def test_initialization(self, embedding_generator):
        """Test embedding generator initialization"""
        assert embedding_generator.model_name == 'all-MiniLM-L6-v2'
        assert embedding_generator.use_tfidf is True
        assert embedding_generator.sentence_model is not None
        assert embedding_generator.tfidf_vectorizer is not None
    
    def test_generate_embeddings(self, embedding_generator):
        """Test embedding generation"""
        texts = [
            "VPN connection issue",
            "Database query slow",
            "Software installation error"
        ]
        
        embeddings = embedding_generator.generate_embeddings(texts, show_progress=False)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0  # Should have embedding dimensions
        assert not np.isnan(embeddings).any()
    
    def test_fit_tfidf(self, embedding_generator, sample_tickets):
        """Test TF-IDF fitting"""
        texts = [ticket.full_text for ticket in sample_tickets]
        
        embedding_generator.fit_tfidf(texts)
        
        assert embedding_generator.is_fitted is True
    
    def test_generate_tfidf_features(self, embedding_generator, sample_tickets):
        """Test TF-IDF feature generation"""
        texts = [ticket.full_text for ticket in sample_tickets]
        
        embedding_generator.fit_tfidf(texts)
        tfidf_features = embedding_generator.generate_tfidf_features(texts)
        
        assert tfidf_features.shape[0] == len(texts)
        assert tfidf_features.shape[1] > 0
        assert not np.isnan(tfidf_features).any()
    
    def test_combined_features(self, embedding_generator, sample_tickets):
        """Test combined feature generation"""
        texts = [ticket.full_text for ticket in sample_tickets]
        
        embedding_generator.fit_tfidf(texts)
        combined = embedding_generator.generate_combined_features(texts)
        
        assert combined.shape[0] == len(texts)
        # Should be wider than embeddings alone
        embeddings = embedding_generator.generate_embeddings(texts, show_progress=False)
        assert combined.shape[1] > embeddings.shape[1]


# Tests for TicketClassifier
class TestTicketClassifier:
    
    def test_initialization(self, embedding_generator):
        """Test classifier initialization"""
        classifier = TicketClassifier(
            embedding_generator,
            classifier_type='gradient_boosting'
        )
        
        assert classifier.embedding_generator is not None
        assert classifier.classifier_type == 'gradient_boosting'
        assert classifier.is_trained is False
    
    def test_training(self, embedding_generator, sample_tickets):
        """Test classifier training"""
        # Need more tickets for proper training
        extended_tickets = sample_tickets * 10  # 50 tickets
        
        classifier = TicketClassifier(embedding_generator, classifier_type='logistic_regression')
        metrics = classifier.train(extended_tickets, validation_split=0.2, use_cross_validation=False)
        
        assert classifier.is_trained is True
        assert 'validation_accuracy' in metrics
        assert 0 <= metrics['validation_accuracy'] <= 1
    
    def test_prediction(self, embedding_generator, sample_tickets):
        """Test ticket prediction"""
        extended_tickets = sample_tickets * 10
        
        classifier = TicketClassifier(embedding_generator, classifier_type='logistic_regression')
        classifier.train(extended_tickets, validation_split=0.2, use_cross_validation=False)
        
        # Predict single ticket
        test_ticket = sample_tickets[0]
        prediction = classifier.predict(test_ticket)
        
        assert isinstance(prediction, str)
        assert prediction in [t.category for t in sample_tickets]
    
    def test_prediction_with_probabilities(self, embedding_generator, sample_tickets):
        """Test prediction with probability scores"""
        extended_tickets = sample_tickets * 10
        
        classifier = TicketClassifier(embedding_generator, classifier_type='logistic_regression')
        classifier.train(extended_tickets, validation_split=0.2, use_cross_validation=False)
        
        test_ticket = sample_tickets[0]
        prediction, probs = classifier.predict(test_ticket, return_probabilities=True)
        
        assert isinstance(prediction, str)
        assert len(probs) == len(classifier.label_encoder.classes_)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    
    def test_batch_prediction(self, embedding_generator, sample_tickets):
        """Test batch ticket prediction"""
        extended_tickets = sample_tickets * 10
        
        classifier = TicketClassifier(embedding_generator, classifier_type='logistic_regression')
        classifier.train(extended_tickets, validation_split=0.2, use_cross_validation=False)
        
        predictions = classifier.predict(sample_tickets[:3])
        
        assert isinstance(predictions, list)
        assert len(predictions) == 3


# Tests for KnowledgeBaseRecommender
class TestKnowledgeBaseRecommender:
    
    def test_initialization(self, embedding_generator):
        """Test recommender initialization"""
        recommender = KnowledgeBaseRecommender(
            embedding_generator,
            similarity_metric='cosine'
        )
        
        assert recommender.embedding_generator is not None
        assert recommender.similarity_metric == 'cosine'
        assert recommender.is_indexed is False
    
    def test_indexing(self, embedding_generator, sample_kb_articles):
        """Test article indexing"""
        recommender = KnowledgeBaseRecommender(embedding_generator)
        recommender.index_articles(sample_kb_articles)
        
        assert recommender.is_indexed is True
        assert len(recommender.articles) == len(sample_kb_articles)
        assert recommender.article_embeddings.shape[0] == len(sample_kb_articles)
    
    def test_recommendation(self, embedding_generator, sample_tickets, sample_kb_articles):
        """Test article recommendation"""
        recommender = KnowledgeBaseRecommender(embedding_generator)
        recommender.index_articles(sample_kb_articles)
        
        ticket = sample_tickets[0]  # VPN ticket
        recommendations = recommender.recommend(ticket, top_k=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(score, (float, np.float32, np.float64)) 
                  for _, score in recommendations)
        
        # Check if VPN article is recommended (should be most similar)
        recommended_ids = [article.article_id for article, _ in recommendations]
        assert "KB-001" in recommended_ids  # VPN article
    
    def test_category_filtering(self, embedding_generator, sample_tickets, sample_kb_articles):
        """Test category-filtered recommendations"""
        recommender = KnowledgeBaseRecommender(embedding_generator)
        recommender.index_articles(sample_kb_articles)
        
        ticket = sample_tickets[0]  # Network ticket
        recommendations = recommender.recommend(
            ticket, 
            top_k=5, 
            category_filter="Network"
        )
        
        # All recommendations should be from Network category
        for article, _ in recommendations:
            assert article.category == "Network"
    
    def test_batch_recommendation(self, embedding_generator, sample_tickets, sample_kb_articles):
        """Test batch article recommendation"""
        recommender = KnowledgeBaseRecommender(embedding_generator)
        recommender.index_articles(sample_kb_articles)
        
        recommendations = recommender.batch_recommend(sample_tickets[:3], top_k=2)
        
        assert len(recommendations) == 3
        assert all(len(recs) <= 2 for recs in recommendations)


# Tests for PerformanceEvaluator
class TestPerformanceEvaluator:
    
    def test_classifier_evaluation(self, embedding_generator, sample_tickets):
        """Test classifier performance evaluation"""
        extended_tickets = sample_tickets * 20  # 100 tickets
        split_idx = int(len(extended_tickets) * 0.8)
        train_tickets = extended_tickets[:split_idx]
        test_tickets = extended_tickets[split_idx:]
        
        classifier = TicketClassifier(embedding_generator, classifier_type='logistic_regression')
        classifier.train(train_tickets, validation_split=0.2, use_cross_validation=False)
        
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_classifier(classifier, test_tickets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_recommender_evaluation(self, embedding_generator, sample_tickets, sample_kb_articles):
        """Test recommender performance evaluation"""
        recommender = KnowledgeBaseRecommender(embedding_generator)
        recommender.index_articles(sample_kb_articles)
        
        # Create ground truth mapping
        ground_truth = {
            ticket.ticket_id: [f"KB-00{i+1}" for i, t in enumerate(sample_tickets) 
                              if t.category == ticket.category]
            for ticket in sample_tickets
        }
        
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_recommender(
            recommender,
            sample_tickets,
            ground_truth,
            top_k=3
        )
        
        assert 'precision@3' in metrics
        assert 'ndcg@3' in metrics
        assert 'mrr' in metrics
        assert 0 <= metrics['precision@3'] <= 1
    
    def test_resolution_time_evaluation(self):
        """Test resolution time impact evaluation"""
        baseline_tickets = [
            TicketData("T-1", "Title", "Desc", "Cat", "Med", 5.0),
            TicketData("T-2", "Title", "Desc", "Cat", "Med", 6.0),
            TicketData("T-3", "Title", "Desc", "Cat", "Med", 4.5),
        ]
        
        improved_tickets = [
            TicketData("T-1", "Title", "Desc", "Cat", "Med", 3.5),
            TicketData("T-2", "Title", "Desc", "Cat", "Med", 4.2),
            TicketData("T-3", "Title", "Desc", "Cat", "Med", 3.0),
        ]
        
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_resolution_time_impact(
            baseline_tickets,
            improved_tickets
        )
        
        assert 'baseline_mean_hours' in metrics
        assert 'improved_mean_hours' in metrics
        assert 'reduction_percentage' in metrics
        assert metrics['reduction_percentage'] > 0  # Should show improvement


# Tests for TechnicalSupportSystem (Integration Tests)
class TestTechnicalSupportSystem:
    
    def test_initialization(self):
        """Test system initialization"""
        system = TechnicalSupportSystem(
            embedding_model='all-MiniLM-L6-v2',
            classifier_type='logistic_regression',
            use_tfidf=True
        )
        
        assert system.embedding_generator is not None
        assert system.classifier is not None
        assert system.recommender is not None
        assert system.evaluator is not None
    
    def test_training(self, sample_tickets, sample_kb_articles):
        """Test complete system training"""
        extended_tickets = sample_tickets * 20
        
        system = TechnicalSupportSystem(
            classifier_type='logistic_regression'
        )
        
        report = system.train(
            extended_tickets,
            sample_kb_articles,
            validation_split=0.2
        )
        
        assert 'classifier_metrics' in report
        assert 'knowledge_base_size' in report
        assert report['knowledge_base_size'] == len(sample_kb_articles)
    
    def test_process_ticket(self, sample_tickets, sample_kb_articles):
        """Test ticket processing"""
        extended_tickets = sample_tickets * 20
        
        system = TechnicalSupportSystem(classifier_type='logistic_regression')
        system.train(extended_tickets, sample_kb_articles, validation_split=0.2)
        
        result = system.process_ticket(sample_tickets[0], recommend_articles=True, top_k=3)
        
        assert 'ticket_id' in result
        assert 'predicted_category' in result
        assert 'confidence' in result
        assert 'recommendations' in result
        assert len(result['recommendations']) <= 3
    
    def test_save_load(self, tmp_path, sample_tickets, sample_kb_articles):
        """Test model saving and loading"""
        extended_tickets = sample_tickets * 20
        
        # Train and save
        system = TechnicalSupportSystem(classifier_type='logistic_regression')
        system.train(extended_tickets, sample_kb_articles, validation_split=0.2)
        
        save_dir = tmp_path / "test_model"
        system.save(str(save_dir))
        
        # Load
        loaded_system = TechnicalSupportSystem.load(str(save_dir))
        
        # Test loaded system
        test_ticket = sample_tickets[0]
        result = loaded_system.process_ticket(test_ticket)
        
        assert 'predicted_category' in result
        assert result['predicted_category'] in [t.category for t in sample_tickets]


# Tests for Data Classes
class TestDataClasses:
    
    def test_ticket_data_full_text(self):
        """Test TicketData full_text property"""
        ticket = TicketData(
            ticket_id="T-001",
            title="Test Title",
            description="Test Description",
            category="Network",
            priority="High",
            resolution_time=5.0
        )
        
        assert ticket.full_text == "Test Title. Test Description"
    
    def test_kb_article_full_text(self):
        """Test KnowledgeBaseArticle full_text property"""
        article = KnowledgeBaseArticle(
            article_id="KB-001",
            title="Test Article",
            content="Test Content",
            category="Network",
            tags=["tag1", "tag2"]
        )
        
        expected = "Test Article. Test Content. Tags: tag1 tag2"
        assert article.full_text == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

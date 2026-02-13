# Technical Support Ticket Classification and Knowledge Base Recommendation System

## Overview

This production-ready system provides intelligent technical support ticket classification and knowledge base article recommendations using domain-adapted language models and machine learning.

## Key Features

### 1. **Multi-Class Ticket Classification**
- Automatic routing of tickets to appropriate support categories
- Confidence scores for predictions
- Support for 8+ technical categories (Network, Hardware, Software, Database, Security, Email, Cloud, Account)
- Ensemble classification with gradient boosting

### 2. **Semantic Knowledge Base Recommendations**
- Vector similarity-based article matching
- Category-filtered recommendations
- Batch processing for multiple tickets
- Relevance scoring using cosine similarity

### 3. **Domain-Adapted Embeddings**
- Sentence-BERT transformers for semantic understanding
- Combined TF-IDF and neural embeddings
- Optimized for technical documentation

### 4. **Performance Evaluation**
- Classification accuracy metrics
- Recommendation relevance (Precision@K, NDCG, MRR)
- Resolution time impact analysis
- Per-category performance tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface / API                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              TechnicalSupportSystem                          │
│  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │  TicketClassifier    │  │ KnowledgeBaseRecommender   │  │
│  │  - Gradient Boosting │  │ - Vector Similarity        │  │
│  │  - Random Forest     │  │ - Category Filtering       │  │
│  │  - Label Encoding    │  │ - Batch Processing         │  │
│  └──────────┬───────────┘  └────────────┬───────────────┘  │
│             └──────────┬────────────────┘                   │
│                        │                                     │
│              ┌─────────▼──────────┐                         │
│              │ EmbeddingGenerator │                         │
│              │ - SentenceTransf.  │                         │
│              │ - TF-IDF Features  │                         │
│              │ - Feature Combo    │                         │
│              └────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Clone or Download the Repository

```bash
# Create project directory
mkdir ticket_classification_system
cd ticket_classification_system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run will download the sentence-transformers model (~90MB) which will be cached for future use.

### 3. Verify Installation

```python
from ticket_classifier import TechnicalSupportSystem
print("Installation successful!")
```

## Quick Start

### Training a New Model

```python
from ticket_classifier import (
    TechnicalSupportSystem,
    TicketData,
    KnowledgeBaseArticle
)

# Prepare training data
tickets = [
    TicketData(
        ticket_id="T-001",
        title="VPN connection failing",
        description="Cannot connect to corporate VPN. Getting timeout errors.",
        category="Network",
        priority="High",
        resolution_time=3.5
    ),
    # Add more tickets...
]

kb_articles = [
    KnowledgeBaseArticle(
        article_id="KB-001",
        title="VPN Troubleshooting Guide",
        content="Steps to diagnose and fix VPN connectivity issues...",
        category="Network",
        tags=["vpn", "connection", "troubleshoot"]
    ),
    # Add more articles...
]

# Initialize and train system
system = TechnicalSupportSystem(
    embedding_model='all-MiniLM-L6-v2',
    classifier_type='gradient_boosting',
    use_tfidf=True
)

training_report = system.train(
    training_tickets=tickets,
    knowledge_base=kb_articles,
    validation_split=0.2
)

print(f"Validation Accuracy: {training_report['classifier_metrics']['validation_accuracy']:.4f}")

# Save trained model
system.save('./my_trained_model')
```

### Loading and Using a Trained Model

```python
# Load previously trained model
system = TechnicalSupportSystem.load('./my_trained_model')

# Process a new ticket
new_ticket = TicketData(
    ticket_id="T-999",
    title="Database query timeout",
    description="Production queries running very slow. Timeouts occurring frequently.",
    category="Unknown",  # Will be predicted
    priority="High",
    resolution_time=0.0
)

result = system.process_ticket(new_ticket, recommend_articles=True, top_k=5)

print(f"Predicted Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nRecommended Articles:")
for i, rec in enumerate(result['recommendations'], 1):
    print(f"{i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")
```

### Running the Demo

```bash
python demo.py
```

This will:
1. Generate synthetic training data (1,200 tickets across 8 categories)
2. Train the classification and recommendation models
3. Evaluate system performance
4. Demonstrate real-time ticket processing
5. Save the trained model

**Expected Output:**
```
Classification Accuracy: 92-96%
Weighted F1-Score: 92-96%
Recommendation Precision@5: 85-90%
Resolution Time Reduction: 20-30%
```

## API Server Deployment

### Starting the Server

```bash
# Basic usage
python api_server.py

# Custom configuration
python api_server.py --model-path ./trained_model --port 8080 --host 0.0.0.0
```

### API Endpoints

#### 1. Health Check
```bash
GET /health

Response:
{
    "status": "healthy",
    "timestamp": "2025-02-13T10:30:00",
    "system_loaded": true
}
```

#### 2. Classify Ticket
```bash
POST /api/v1/classify
Content-Type: application/json

{
    "ticket_id": "T-12345",
    "title": "Cannot access email",
    "description": "Getting authentication errors when trying to login to Outlook",
    "priority": "Medium"
}

Response:
{
    "ticket_id": "T-12345",
    "predicted_category": "Email",
    "confidence": 0.89,
    "top_predictions": [
        {"category": "Email", "confidence": 0.89},
        {"category": "Account", "confidence": 0.07},
        {"category": "Security", "confidence": 0.03}
    ],
    "processing_time_ms": 45.2
}
```

#### 3. Recommend Articles
```bash
POST /api/v1/recommend
Content-Type: application/json

{
    "ticket_id": "T-12345",
    "title": "Cannot access email",
    "description": "Getting authentication errors...",
    "category": "Email",  # Optional
    "top_k": 5
}

Response:
{
    "ticket_id": "T-12345",
    "category": "Email",
    "recommendations": [
        {
            "article_id": "KB-0045",
            "title": "Email Authentication Issues",
            "category": "Email",
            "similarity_score": 0.87,
            "tags": ["email", "authentication", "outlook"]
        }
    ],
    "processing_time_ms": 52.1
}
```

#### 4. Complete Processing
```bash
POST /api/v1/process
Content-Type: application/json

{
    "ticket_id": "T-12345",
    "title": "Cannot access email",
    "description": "Getting authentication errors...",
    "top_k": 5
}

Response: (Combined classification + recommendations)
```

#### 5. Batch Processing
```bash
POST /api/v1/batch/process
Content-Type: application/json

{
    "tickets": [
        {"ticket_id": "T-001", "title": "...", "description": "..."},
        {"ticket_id": "T-002", "title": "...", "description": "..."}
    ],
    "top_k": 5
}
```

#### 6. Get Categories
```bash
GET /api/v1/categories

Response:
{
    "categories": ["Network", "Hardware", "Software", ...],
    "count": 8
}
```

#### 7. System Statistics
```bash
GET /api/v1/stats

Response:
{
    "model_info": {
        "trained_at": "2025-02-13T09:15:00",
        "categories": [...],
        "num_categories": 8
    },
    "training_metrics": {
        "validation_accuracy": 0.94,
        ...
    },
    "knowledge_base": {
        "total_articles": 24,
        "indexed": true
    }
}
```

### Production Deployment

#### Using Gunicorn (Recommended)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 worker processes
gunicorn -w 4 -b 0.0.0.0:8000 api_server:app

# With configuration
gunicorn -c gunicorn_config.py api_server:app
```

**gunicorn_config.py**:
```python
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
accesslog = "./logs/access.log"
errorlog = "./logs/error.log"
loglevel = "info"
```

#### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "--timeout", "120", "api_server:app"]
```

Build and run:
```bash
docker build -t ticket-classifier .
docker run -p 8000:8000 -v $(pwd)/trained_model:/app/trained_model ticket-classifier
```

## System Components

### EmbeddingGenerator

Generates vector representations of text using:
- **Sentence-BERT**: Pre-trained transformer models for semantic similarity
- **TF-IDF**: Term frequency features for keyword matching
- **Combined Features**: Weighted combination of both approaches

**Configuration Options:**
```python
generator = EmbeddingGenerator(
    model_name='all-MiniLM-L6-v2',  # or 'all-mpnet-base-v2' for better quality
    use_tfidf=True,                  # Enable TF-IDF features
    cache_dir='./model_cache'        # Model cache location
)
```

### TicketClassifier

Multi-class classification with support for:
- **Gradient Boosting** (default, best accuracy)
- **Random Forest** (good interpretability)
- **Logistic Regression** (fast, baseline)

**Key Methods:**
- `train()`: Train on historical tickets
- `predict()`: Classify new tickets
- `get_feature_importance()`: Analyze important features (tree-based only)

### KnowledgeBaseRecommender

Semantic article recommendations using:
- **Cosine Similarity**: Default, best for normalized embeddings
- **Euclidean Distance**: Alternative distance metric
- **Category Filtering**: Restrict to specific categories
- **Batch Processing**: Efficient multi-ticket processing

**Key Methods:**
- `index_articles()`: Build article index
- `recommend()`: Get top-K recommendations
- `batch_recommend()`: Process multiple tickets efficiently

### PerformanceEvaluator

Comprehensive evaluation metrics:

**Classification Metrics:**
- Accuracy
- Precision, Recall, F1-Score (weighted and per-category)
- Confusion Matrix
- Cross-validation scores

**Recommendation Metrics:**
- Precision@K, Recall@K
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Reciprocal Rank (MRR)

**Business Metrics:**
- Average resolution time
- Resolution time reduction percentage
- Median resolution time

## Performance Optimization

### 1. Embedding Caching

```python
# Cache embeddings for frequently accessed articles
import joblib

# After indexing
embeddings_cache = {
    'articles': system.recommender.article_embeddings,
    'article_ids': [a.article_id for a in system.recommender.articles]
}
joblib.dump(embeddings_cache, 'embeddings_cache.pkl')

# Load cached embeddings
cached = joblib.load('embeddings_cache.pkl')
```

### 2. Batch Processing

Process multiple tickets efficiently:

```python
# Instead of:
for ticket in tickets:
    result = system.process_ticket(ticket)

# Use batch methods:
recommendations = system.recommender.batch_recommend(tickets, top_k=5)
```

### 3. Model Selection

For different requirements:

```python
# Fast inference, lower accuracy
system = TechnicalSupportSystem(
    embedding_model='all-MiniLM-L6-v2',
    classifier_type='logistic_regression'
)

# Best accuracy, slower inference
system = TechnicalSupportSystem(
    embedding_model='all-mpnet-base-v2',
    classifier_type='gradient_boosting'
)
```

### 4. Hardware Acceleration

For GPU acceleration (if available):

```python
import torch

# EmbeddingGenerator will automatically use GPU if available
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

## Monitoring and Maintenance

### Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Monitoring

Track key metrics in production:

```python
from datetime import datetime
import json

def log_prediction(ticket_id, predicted_category, confidence, processing_time):
    """Log predictions for monitoring"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'ticket_id': ticket_id,
        'predicted_category': predicted_category,
        'confidence': confidence,
        'processing_time_ms': processing_time
    }
    
    with open('predictions.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

### Model Retraining

Retrain periodically with new data:

```python
# Load existing model
system = TechnicalSupportSystem.load('./trained_model')

# Add new training data
new_tickets = load_recent_tickets()

# Combine with original data if needed
all_tickets = original_tickets + new_tickets

# Retrain
system.train(all_tickets, kb_articles, validation_split=0.2)
system.save('./trained_model_v2')
```

## Advanced Features

### Custom Categories

```python
# Define custom categories for your organization
CUSTOM_CATEGORIES = {
    'SalesForce': {...},
    'HR_Systems': {...},
    'Payroll': {...}
}

# Use same training approach with custom data
```

### Multi-Language Support

```python
# Use multilingual models
system = TechnicalSupportSystem(
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2'
)
```

### Integration with Ticketing Systems

```python
# Example: Integration with JIRA, ServiceNow, Zendesk

def process_new_ticket_from_jira(jira_ticket):
    """Process ticket from JIRA"""
    ticket = TicketData(
        ticket_id=jira_ticket['key'],
        title=jira_ticket['fields']['summary'],
        description=jira_ticket['fields']['description'],
        category='Unknown',
        priority=jira_ticket['fields']['priority']['name'],
        resolution_time=0.0
    )
    
    result = system.process_ticket(ticket)
    
    # Update JIRA with classification
    jira_client.update_ticket(
        ticket_id=jira_ticket['key'],
        category=result['predicted_category'],
        recommended_articles=result['recommendations']
    )
    
    return result
```

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```python
# Reduce batch size
embeddings = generator.generate_embeddings(texts, batch_size=16)

# Or process in chunks
for chunk in chunks(texts, 100):
    embeddings = generator.generate_embeddings(chunk)
```

**2. Slow Inference**
```python
# Use smaller model
system = TechnicalSupportSystem(embedding_model='all-MiniLM-L6-v2')

# Disable TF-IDF
system = TechnicalSupportSystem(use_tfidf=False)
```

**3. Low Accuracy**
```python
# Use larger model
system = TechnicalSupportSystem(embedding_model='all-mpnet-base-v2')

# Increase training data
# Add more diverse examples per category

# Try different classifier
system = TechnicalSupportSystem(classifier_type='gradient_boosting')
```

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, questions, or contributions, please refer to the project documentation or contact your system administrator.

## Citation

If you use this system in your research or production environment, please cite:

```
Technical Support Ticket Classification and Knowledge Base Recommendation System
Version 1.0 (2025)
```

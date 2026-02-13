"""
Configuration file for Technical Support System

This file contains all configurable parameters for the system.
Modify these settings based on your environment and requirements.
"""

import os
from pathlib import Path

# ============================================================================
# Model Configuration
# ============================================================================

# Sentence Transformer Model
# Options: 'all-MiniLM-L6-v2' (fast, good), 'all-mpnet-base-v2' (slower, better)
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Classifier Type
# Options: 'gradient_boosting', 'random_forest', 'logistic_regression'
CLASSIFIER_TYPE = 'gradient_boosting'

# Use TF-IDF features in addition to embeddings
USE_TFIDF = True

# TF-IDF weight in combined features (0.0 to 1.0)
TFIDF_WEIGHT = 0.3

# Similarity metric for recommendations
# Options: 'cosine', 'euclidean'
SIMILARITY_METRIC = 'cosine'

# ============================================================================
# Training Configuration
# ============================================================================

# Validation split ratio
VALIDATION_SPLIT = 0.2

# Use cross-validation during training
USE_CROSS_VALIDATION = True

# Number of cross-validation folds
CV_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Classifier Parameters
# ============================================================================

# Gradient Boosting
GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'subsample': 0.8,
    'random_state': RANDOM_SEED
}

# Random Forest
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Logistic Regression
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# ============================================================================
# TF-IDF Parameters
# ============================================================================

TFIDF_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 3),
    'min_df': 2,
    'max_df': 0.8,
    'stop_words': 'english'
}

# ============================================================================
# Recommendation Configuration
# ============================================================================

# Default number of recommendations
DEFAULT_TOP_K = 5

# Minimum similarity threshold for recommendations
MIN_SIMILARITY_THRESHOLD = 0.0

# Filter recommendations by predicted category
FILTER_BY_CATEGORY = True

# ============================================================================
# API Server Configuration
# ============================================================================

# Server settings
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = False

# Request limits
MAX_BATCH_SIZE = 100
MAX_REQUEST_SIZE_MB = 10

# Timeout settings (seconds)
REQUEST_TIMEOUT = 120

# CORS settings
CORS_ORIGINS = ['*']  # Change to specific origins in production

# ============================================================================
# Performance Configuration
# ============================================================================

# Embedding batch size
EMBEDDING_BATCH_SIZE = 32

# Show progress bars
SHOW_PROGRESS = True

# Number of workers for parallel processing
N_WORKERS = -1  # -1 means use all available cores

# ============================================================================
# Storage Configuration
# ============================================================================

# Base directory for models and data
BASE_DIR = Path(__file__).parent

# Model storage directory
MODEL_DIR = BASE_DIR / 'models'

# Cache directory for embeddings and models
CACHE_DIR = BASE_DIR / 'cache'

# Log directory
LOG_DIR = BASE_DIR / 'logs'

# Data directory
DATA_DIR = BASE_DIR / 'data'

# Create directories if they don't exist
for directory in [MODEL_DIR, CACHE_DIR, LOG_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

LOG_FILE = LOG_DIR / 'system.log'

# Log rotation
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Metrics to compute during evaluation
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'classification_report'
]

# Recommendation metrics
RECOMMENDATION_METRICS = [
    'precision@k',
    'recall@k',
    'ndcg@k',
    'mrr'
]

# Values of K for recommendation metrics
RECOMMENDATION_K_VALUES = [1, 3, 5, 10]

# ============================================================================
# Production Configuration
# ============================================================================

# Enable production mode
PRODUCTION_MODE = os.getenv('PRODUCTION_MODE', 'false').lower() == 'true'

# Database connection (if using external storage)
DATABASE_URL = os.getenv('DATABASE_URL', None)

# Redis cache (if using caching)
REDIS_URL = os.getenv('REDIS_URL', None)

# Monitoring and alerting
ENABLE_MONITORING = PRODUCTION_MODE
ENABLE_METRICS_EXPORT = PRODUCTION_MODE

# Alert thresholds
ALERT_THRESHOLDS = {
    'low_confidence': 0.5,  # Alert if prediction confidence < 50%
    'high_processing_time': 1000,  # Alert if processing > 1000ms
    'error_rate': 0.05  # Alert if error rate > 5%
}

# ============================================================================
# Feature Flags
# ============================================================================

FEATURE_FLAGS = {
    'enable_batch_processing': True,
    'enable_category_filtering': True,
    'enable_confidence_scores': True,
    'enable_article_feedback': True,
    'enable_auto_retraining': False,
    'enable_a_b_testing': False
}

# ============================================================================
# Categories Configuration
# ============================================================================

# Define your ticket categories
# This is used for validation and documentation
TICKET_CATEGORIES = [
    'Network',
    'Hardware',
    'Software',
    'Database',
    'Security',
    'Email',
    'Cloud',
    'Account'
]

# Category priorities (for routing SLA)
CATEGORY_PRIORITIES = {
    'Security': 1,  # Highest priority
    'Database': 2,
    'Network': 3,
    'Cloud': 3,
    'Hardware': 4,
    'Software': 4,
    'Email': 5,
    'Account': 5
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_model_path(model_name: str = 'default') -> Path:
    """Get path for saved model"""
    return MODEL_DIR / model_name


def get_cache_path(cache_name: str) -> Path:
    """Get path for cache file"""
    return CACHE_DIR / cache_name


def get_log_path(log_name: str = 'system.log') -> Path:
    """Get path for log file"""
    return LOG_DIR / log_name


def get_data_path(data_name: str) -> Path:
    """Get path for data file"""
    return DATA_DIR / data_name


# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check validation split
    if not 0 < VALIDATION_SPLIT < 1:
        errors.append("VALIDATION_SPLIT must be between 0 and 1")
    
    # Check TF-IDF weight
    if not 0 <= TFIDF_WEIGHT <= 1:
        errors.append("TFIDF_WEIGHT must be between 0 and 1")
    
    # Check classifier type
    valid_classifiers = ['gradient_boosting', 'random_forest', 'logistic_regression']
    if CLASSIFIER_TYPE not in valid_classifiers:
        errors.append(f"CLASSIFIER_TYPE must be one of {valid_classifiers}")
    
    # Check similarity metric
    valid_metrics = ['cosine', 'euclidean']
    if SIMILARITY_METRIC not in valid_metrics:
        errors.append(f"SIMILARITY_METRIC must be one of {valid_metrics}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Validate on import
if __name__ != '__main__':
    validate_config()


if __name__ == '__main__':
    """Print configuration summary when run directly"""
    print("=" * 80)
    print("Technical Support System Configuration")
    print("=" * 80)
    print()
    
    print("Model Configuration:")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Classifier Type: {CLASSIFIER_TYPE}")
    print(f"  Use TF-IDF: {USE_TFIDF}")
    print(f"  Similarity Metric: {SIMILARITY_METRIC}")
    print()
    
    print("Training Configuration:")
    print(f"  Validation Split: {VALIDATION_SPLIT}")
    print(f"  Cross-Validation: {USE_CROSS_VALIDATION}")
    print(f"  CV Folds: {CV_FOLDS}")
    print()
    
    print("API Configuration:")
    print(f"  Host: {API_HOST}")
    print(f"  Port: {API_PORT}")
    print(f"  Debug Mode: {API_DEBUG}")
    print()
    
    print("Storage:")
    print(f"  Model Directory: {MODEL_DIR}")
    print(f"  Cache Directory: {CACHE_DIR}")
    print(f"  Log Directory: {LOG_DIR}")
    print(f"  Data Directory: {DATA_DIR}")
    print()
    
    print("Categories:")
    for cat in TICKET_CATEGORIES:
        priority = CATEGORY_PRIORITIES.get(cat, 'N/A')
        print(f"  {cat}: Priority {priority}")
    print()
    
    print("Production Mode:", "ENABLED" if PRODUCTION_MODE else "DISABLED")
    print()
    
    # Validate configuration
    try:
        validate_config()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration errors:\n{e}")

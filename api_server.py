"""
Production API Server for Technical Support System

This module provides a REST API interface for the ticket classification
and knowledge base recommendation system using Flask.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, List, Optional
from datetime import datetime
import traceback

from ticket_classifier import (
    TechnicalSupportSystem,
    TicketData
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global system instance
support_system: Optional[TechnicalSupportSystem] = None


def init_system(model_path: str = './trained_model'):
    """Initialize or load the support system"""
    global support_system
    try:
        logger.info(f"Loading support system from {model_path}")
        support_system = TechnicalSupportSystem.load(model_path)
        logger.info("Support system loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load support system: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_loaded': support_system is not None
    })


@app.route('/api/v1/classify', methods=['POST'])
def classify_ticket():
    """
    Classify a support ticket
    
    Request JSON:
    {
        "ticket_id": "TICKET-12345",
        "title": "Cannot connect to VPN",
        "description": "Getting timeout errors...",
        "priority": "High"  // Optional
    }
    
    Response JSON:
    {
        "ticket_id": "TICKET-12345",
        "predicted_category": "Network",
        "confidence": 0.95,
        "top_predictions": [
            {"category": "Network", "confidence": 0.95},
            {"category": "Security", "confidence": 0.03}
        ],
        "processing_time_ms": 45.2
    }
    """
    try:
        if support_system is None:
            return jsonify({'error': 'Support system not initialized'}), 503
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['ticket_id', 'title', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        start_time = datetime.now()
        
        # Create ticket object
        ticket = TicketData(
            ticket_id=data['ticket_id'],
            title=data['title'],
            description=data['description'],
            category='Unknown',  # Will be predicted
            priority=data.get('priority', 'Medium'),
            resolution_time=0.0
        )
        
        # Get predictions
        predicted_category, probabilities = support_system.classifier.predict(
            ticket,
            return_probabilities=True
        )
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[::-1][:3]
        top_predictions = [
            {
                'category': support_system.classifier.label_encoder.classes_[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return jsonify({
            'ticket_id': data['ticket_id'],
            'predicted_category': predicted_category,
            'confidence': float(probabilities.max()),
            'top_predictions': top_predictions,
            'processing_time_ms': processing_time
        })
    
    except Exception as e:
        logger.error(f"Error in classify_ticket: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/recommend', methods=['POST'])
def recommend_articles():
    """
    Recommend knowledge base articles for a ticket
    
    Request JSON:
    {
        "ticket_id": "TICKET-12345",
        "title": "Cannot connect to VPN",
        "description": "Getting timeout errors...",
        "category": "Network",  // Optional, will be predicted if not provided
        "top_k": 5  // Optional, default 5
    }
    
    Response JSON:
    {
        "ticket_id": "TICKET-12345",
        "category": "Network",
        "recommendations": [
            {
                "article_id": "KB-0001",
                "title": "VPN Troubleshooting Guide",
                "category": "Network",
                "similarity_score": 0.87,
                "tags": ["vpn", "connection", "troubleshoot"]
            }
        ],
        "processing_time_ms": 52.1
    }
    """
    try:
        if support_system is None:
            return jsonify({'error': 'Support system not initialized'}), 503
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['ticket_id', 'title', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        start_time = datetime.now()
        
        top_k = data.get('top_k', 5)
        category = data.get('category')
        
        # Create ticket object
        ticket = TicketData(
            ticket_id=data['ticket_id'],
            title=data['title'],
            description=data['description'],
            category=category or 'Unknown',
            priority=data.get('priority', 'Medium'),
            resolution_time=0.0
        )
        
        # If category not provided, predict it
        if not category:
            category = support_system.classifier.predict(ticket)
        
        # Get recommendations
        recommendations = support_system.recommender.recommend(
            ticket,
            top_k=top_k,
            category_filter=category
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return jsonify({
            'ticket_id': data['ticket_id'],
            'category': category,
            'recommendations': [
                {
                    'article_id': article.article_id,
                    'title': article.title,
                    'category': article.category,
                    'similarity_score': float(score),
                    'tags': article.tags,
                    'resolution_count': article.resolution_count
                }
                for article, score in recommendations
            ],
            'processing_time_ms': processing_time
        })
    
    except Exception as e:
        logger.error(f"Error in recommend_articles: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/process', methods=['POST'])
def process_ticket():
    """
    Complete ticket processing: classification + recommendations
    
    Request JSON:
    {
        "ticket_id": "TICKET-12345",
        "title": "Cannot connect to VPN",
        "description": "Getting timeout errors...",
        "priority": "High",  // Optional
        "top_k": 5  // Optional, default 5
    }
    
    Response JSON:
    {
        "ticket_id": "TICKET-12345",
        "predicted_category": "Network",
        "confidence": 0.95,
        "top_predictions": [...],
        "recommendations": [...],
        "processing_time_ms": 67.3
    }
    """
    try:
        if support_system is None:
            return jsonify({'error': 'Support system not initialized'}), 503
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['ticket_id', 'title', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        start_time = datetime.now()
        
        top_k = data.get('top_k', 5)
        
        # Create ticket object
        ticket = TicketData(
            ticket_id=data['ticket_id'],
            title=data['title'],
            description=data['description'],
            category='Unknown',
            priority=data.get('priority', 'Medium'),
            resolution_time=0.0
        )
        
        # Process ticket
        result = support_system.process_ticket(
            ticket,
            recommend_articles=True,
            top_k=top_k
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result['processing_time_ms'] = processing_time
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in process_ticket: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/batch/process', methods=['POST'])
def batch_process():
    """
    Batch process multiple tickets
    
    Request JSON:
    {
        "tickets": [
            {
                "ticket_id": "TICKET-001",
                "title": "...",
                "description": "..."
            },
            ...
        ],
        "top_k": 5  // Optional
    }
    
    Response JSON:
    {
        "results": [...],
        "total_processed": 10,
        "processing_time_ms": 234.5
    }
    """
    try:
        if support_system is None:
            return jsonify({'error': 'Support system not initialized'}), 503
        
        data = request.get_json()
        
        if 'tickets' not in data or not isinstance(data['tickets'], list):
            return jsonify({'error': 'Missing or invalid tickets array'}), 400
        
        start_time = datetime.now()
        top_k = data.get('top_k', 5)
        
        results = []
        for ticket_data in data['tickets']:
            try:
                ticket = TicketData(
                    ticket_id=ticket_data['ticket_id'],
                    title=ticket_data['title'],
                    description=ticket_data['description'],
                    category='Unknown',
                    priority=ticket_data.get('priority', 'Medium'),
                    resolution_time=0.0
                )
                
                result = support_system.process_ticket(ticket, recommend_articles=True, top_k=top_k)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing ticket {ticket_data.get('ticket_id')}: {e}")
                results.append({
                    'ticket_id': ticket_data.get('ticket_id'),
                    'error': str(e)
                })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'processing_time_ms': processing_time
        })
    
    except Exception as e:
        logger.error(f"Error in batch_process: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/categories', methods=['GET'])
def get_categories():
    """Get list of available ticket categories"""
    try:
        if support_system is None:
            return jsonify({'error': 'Support system not initialized'}), 503
        
        categories = support_system.classifier.label_encoder.classes_.tolist()
        
        return jsonify({
            'categories': categories,
            'count': len(categories)
        })
    
    except Exception as e:
        logger.error(f"Error in get_categories: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        if support_system is None:
            return jsonify({'error': 'Support system not initialized'}), 503
        
        metadata = support_system.classifier.training_metadata
        
        return jsonify({
            'model_info': {
                'trained_at': metadata.get('trained_at'),
                'categories': metadata.get('categories', []),
                'num_categories': len(metadata.get('categories', []))
            },
            'training_metrics': metadata.get('metrics', {}),
            'knowledge_base': {
                'total_articles': len(support_system.recommender.articles),
                'indexed': support_system.recommender.is_indexed
            }
        })
    
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'error': str(e)}), 500


def create_app(model_path: str = './trained_model'):
    """Application factory"""
    init_system(model_path)
    return app


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Technical Support System API Server')
    parser.add_argument('--model-path', type=str, default='./trained_model',
                       help='Path to trained model directory')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize system
    init_system(args.model_path)
    
    # Run server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

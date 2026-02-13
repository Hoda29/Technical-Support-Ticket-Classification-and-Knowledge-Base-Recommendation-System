"""
Demonstration script for Technical Support Ticket Classification System

This script demonstrates the complete workflow including:
- Synthetic data generation
- Model training
- Evaluation
- Real-time ticket processing
"""

import numpy as np
import pandas as pd
from typing import List
import random
from datetime import datetime, timedelta

from ticket_classifier import (
    TicketData,
    KnowledgeBaseArticle,
    TechnicalSupportSystem,
    PerformanceEvaluator
)


# Sample categories and their typical characteristics
TICKET_CATEGORIES = {
    'Network': {
        'keywords': ['connection', 'wifi', 'ethernet', 'vpn', 'dns', 'latency', 'packet loss', 
                    'timeout', 'firewall', 'port', 'bandwidth', 'router'],
        'priority_dist': [0.2, 0.5, 0.3],  # low, medium, high
        'avg_resolution_hours': 3.5
    },
    'Hardware': {
        'keywords': ['laptop', 'desktop', 'monitor', 'keyboard', 'mouse', 'printer', 'scanner',
                    'drive', 'disk', 'memory', 'cpu', 'overheating', 'power', 'battery'],
        'priority_dist': [0.3, 0.4, 0.3],
        'avg_resolution_hours': 8.0
    },
    'Software': {
        'keywords': ['application', 'program', 'install', 'update', 'crash', 'error', 'bug',
                    'license', 'activation', 'compatibility', 'version', 'patch'],
        'priority_dist': [0.4, 0.4, 0.2],
        'avg_resolution_hours': 4.5
    },
    'Database': {
        'keywords': ['query', 'table', 'connection', 'timeout', 'slow', 'index', 'backup',
                    'restore', 'replication', 'deadlock', 'transaction', 'migration'],
        'priority_dist': [0.1, 0.4, 0.5],
        'avg_resolution_hours': 6.0
    },
    'Security': {
        'keywords': ['access', 'permission', 'authentication', 'password', 'locked', 'breach',
                    'vulnerability', 'patch', 'encryption', 'certificate', 'malware'],
        'priority_dist': [0.1, 0.3, 0.6],
        'avg_resolution_hours': 5.0
    },
    'Email': {
        'keywords': ['outlook', 'gmail', 'smtp', 'inbox', 'spam', 'attachment', 'send',
                    'receive', 'mailbox', 'quota', 'sync', 'calendar'],
        'priority_dist': [0.5, 0.4, 0.1],
        'avg_resolution_hours': 2.0
    },
    'Cloud': {
        'keywords': ['aws', 'azure', 'storage', 'bucket', 'deployment', 'container', 'kubernetes',
                    'serverless', 'api', 'gateway', 'instance', 'region'],
        'priority_dist': [0.2, 0.5, 0.3],
        'avg_resolution_hours': 7.5
    },
    'Account': {
        'keywords': ['login', 'account', 'profile', 'settings', 'preferences', 'subscription',
                    'billing', 'payment', 'username', 'registration'],
        'priority_dist': [0.6, 0.3, 0.1],
        'avg_resolution_hours': 1.5
    }
}

ISSUE_TEMPLATES = {
    'Network': [
        "Unable to {action} to {resource}. Getting {error} error.",
        "Experiencing {symptom} when accessing {resource}.",
        "{resource} is {symptom}. Need urgent help.",
        "VPN connection {symptom} after {action}.",
        "DNS resolution failing for {resource}.",
    ],
    'Hardware': [
        "{device} is {symptom}. {additional_info}",
        "Replacement needed for {device} due to {symptom}.",
        "{device} not {action} properly.",
        "Physical damage to {device}. {additional_info}",
        "{device} making unusual {symptom}.",
    ],
    'Software': [
        "{application} {symptom} when {action}.",
        "Cannot {action} {application}. Error: {error}",
        "{application} needs {action}.",
        "Compatibility issue with {application} and {resource}.",
        "License activation failed for {application}.",
    ],
    'Database': [
        "Query performance {symptom} on {resource}.",
        "Database connection {symptom}. {additional_info}",
        "Need to {action} database {resource}.",
        "Replication lag on {resource}.",
        "Deadlock detected in {resource}.",
    ],
    'Security': [
        "Access denied to {resource}. {additional_info}",
        "Need {action} for {resource}.",
        "Security {symptom} detected in {resource}.",
        "Password reset required for {resource}.",
        "Suspicious activity on {resource}.",
    ],
    'Email': [
        "Cannot {action} emails. {additional_info}",
        "Mailbox {symptom}. {additional_info}",
        "Email {symptom} after {action}.",
        "Attachment issues with {resource}.",
        "Calendar sync {symptom}.",
    ],
    'Cloud': [
        "Deployment {symptom} on {resource}.",
        "Cannot {action} to {resource}.",
        "{resource} instance {symptom}.",
        "API {symptom} for {resource}.",
        "Storage quota exceeded for {resource}.",
    ],
    'Account': [
        "Cannot {action} to account. {additional_info}",
        "Account {symptom}. Need help.",
        "Need to {action} account {resource}.",
        "Billing issue with {resource}.",
        "Profile settings {symptom}.",
    ]
}

# Filler words for more natural text
ACTIONS = ['connect', 'access', 'update', 'install', 'configure', 'restart', 'login', 
           'upgrade', 'migrate', 'backup', 'restore']
SYMPTOMS = ['not working', 'failing', 'slow', 'unresponsive', 'crashing', 'freezing',
            'timing out', 'not responding', 'intermittent', 'unstable']
RESOURCES = ['server', 'database', 'application', 'network', 'system', 'service',
            'platform', 'environment', 'endpoint']
ERRORS = ['timeout', '404', '500', 'connection refused', 'access denied', 'not found',
          'permission denied', 'invalid credentials']
ADDITIONAL_INFO = [
    'This is affecting multiple users.',
    'Issue started this morning.',
    'Previously working fine.',
    'Urgent - production down.',
    'Intermittent issue.',
    'Affecting all departments.',
]


def generate_synthetic_tickets(n_tickets: int, category: str) -> List[TicketData]:
    """Generate synthetic tickets for a specific category"""
    tickets = []
    config = TICKET_CATEGORIES[category]
    
    for i in range(n_tickets):
        # Generate title
        template = random.choice(ISSUE_TEMPLATES[category])
        
        title = template.format(
            action=random.choice(ACTIONS),
            symptom=random.choice(SYMPTOMS),
            resource=random.choice(RESOURCES),
            error=random.choice(ERRORS),
            additional_info=random.choice(ADDITIONAL_INFO),
            device=random.choice(config['keywords'][:5]),
            application=random.choice(config['keywords'][:5])
        )
        
        # Generate description with category-specific keywords
        keywords_used = random.sample(config['keywords'], k=min(5, len(config['keywords'])))
        description_parts = [
            f"The {random.choice(keywords_used)} is {random.choice(SYMPTOMS)}.",
            f"Attempted to {random.choice(ACTIONS)} but {random.choice(SYMPTOMS)}.",
            f"Error message: {random.choice(ERRORS)}.",
            random.choice(ADDITIONAL_INFO)
        ]
        description = " ".join(random.sample(description_parts, k=3))
        
        # Assign priority based on distribution
        priority = random.choices(['Low', 'Medium', 'High'], weights=config['priority_dist'])[0]
        
        # Generate resolution time (normal distribution around average)
        base_time = config['avg_resolution_hours']
        resolution_time = max(0.5, np.random.normal(base_time, base_time * 0.3))
        
        ticket = TicketData(
            ticket_id=f"{category[:3].upper()}-{i+1:05d}",
            title=title,
            description=description,
            category=category,
            priority=priority,
            resolution_time=resolution_time,
            resolution_notes=f"Resolved using standard {category.lower()} procedures."
        )
        tickets.append(ticket)
    
    return tickets


def generate_knowledge_base() -> List[KnowledgeBaseArticle]:
    """Generate synthetic knowledge base articles"""
    articles = []
    
    kb_templates = {
        'Network': [
            ('VPN Connection Troubleshooting', 'Steps to resolve VPN connectivity issues', 
             ['vpn', 'connection', 'troubleshoot']),
            ('DNS Resolution Issues', 'How to diagnose and fix DNS problems',
             ['dns', 'network', 'resolution']),
            ('Firewall Configuration Guide', 'Configuring firewall rules for applications',
             ['firewall', 'security', 'ports']),
        ],
        'Hardware': [
            ('Laptop Hardware Diagnostics', 'Running hardware diagnostics on laptops',
             ['laptop', 'diagnostics', 'hardware']),
            ('Printer Installation Guide', 'Installing and configuring network printers',
             ['printer', 'install', 'network']),
            ('Monitor Display Problems', 'Fixing common monitor display issues',
             ['monitor', 'display', 'troubleshoot']),
        ],
        'Software': [
            ('Application Installation Issues', 'Resolving software installation problems',
             ['install', 'software', 'application']),
            ('License Activation Steps', 'How to activate software licenses',
             ['license', 'activation', 'software']),
            ('Software Update Procedures', 'Best practices for updating applications',
             ['update', 'patch', 'software']),
        ],
        'Database': [
            ('Database Performance Tuning', 'Optimizing database query performance',
             ['database', 'performance', 'optimization']),
            ('Backup and Restore Procedures', 'Database backup and recovery steps',
             ['backup', 'restore', 'recovery']),
            ('Connection Pool Configuration', 'Configuring database connection pools',
             ['connection', 'pool', 'database']),
        ],
        'Security': [
            ('Access Control Management', 'Managing user permissions and access',
             ['access', 'permissions', 'security']),
            ('Password Reset Procedures', 'How to reset user passwords securely',
             ['password', 'reset', 'security']),
            ('Security Patch Deployment', 'Deploying security updates and patches',
             ['security', 'patch', 'update']),
        ],
        'Email': [
            ('Email Client Configuration', 'Setting up email clients correctly',
             ['email', 'outlook', 'configuration']),
            ('Mailbox Quota Management', 'Managing email storage quotas',
             ['mailbox', 'quota', 'storage']),
            ('Spam Filter Configuration', 'Configuring spam and junk filters',
             ['spam', 'filter', 'email']),
        ],
        'Cloud': [
            ('Cloud Deployment Best Practices', 'Guidelines for cloud deployments',
             ['cloud', 'deployment', 'best-practices']),
            ('Container Orchestration Guide', 'Managing containerized applications',
             ['container', 'kubernetes', 'orchestration']),
            ('Cloud Storage Configuration', 'Setting up cloud storage buckets',
             ['storage', 'cloud', 'bucket']),
        ],
        'Account': [
            ('Account Creation Process', 'Creating new user accounts',
             ['account', 'creation', 'user']),
            ('Billing and Subscriptions', 'Managing billing and subscription info',
             ['billing', 'subscription', 'payment']),
            ('Profile Settings Guide', 'Updating user profile settings',
             ['profile', 'settings', 'preferences']),
        ]
    }
    
    article_id = 1
    for category, templates in kb_templates.items():
        for title, content, tags in templates:
            # Expand content with category keywords
            keywords = TICKET_CATEGORIES[category]['keywords']
            expanded_content = f"{content}. This article covers {', '.join(keywords[:5])}. "
            expanded_content += f"Common issues include {', '.join(SYMPTOMS[:3])}. "
            expanded_content += f"Solutions typically involve {', '.join(ACTIONS[:3])}."
            
            article = KnowledgeBaseArticle(
                article_id=f"KB-{article_id:04d}",
                title=title,
                content=expanded_content,
                category=category,
                tags=tags,
                resolution_count=random.randint(5, 50)
            )
            articles.append(article)
            article_id += 1
    
    return articles


def generate_ground_truth_mapping(
    tickets: List[TicketData],
    articles: List[KnowledgeBaseArticle]
) -> dict:
    """Generate ground truth article relevance for evaluation"""
    ground_truth = {}
    
    # Create category-based mapping
    category_articles = {}
    for article in articles:
        if article.category not in category_articles:
            category_articles[article.category] = []
        category_articles[article.category].append(article.article_id)
    
    # Assign relevant articles to tickets based on category
    for ticket in tickets:
        if ticket.category in category_articles:
            # Main category articles are relevant
            relevant = category_articles[ticket.category].copy()
            
            # Add 1-2 articles from related categories with some probability
            if random.random() > 0.7:
                other_categories = [c for c in category_articles.keys() if c != ticket.category]
                if other_categories:
                    related_cat = random.choice(other_categories)
                    relevant.extend(random.sample(category_articles[related_cat], 
                                                  k=min(1, len(category_articles[related_cat]))))
            
            ground_truth[ticket.ticket_id] = relevant
    
    return ground_truth


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("Technical Support Ticket Classification System - Demo")
    print("=" * 80)
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # ========================================================================
    # Step 1: Generate Synthetic Data
    # ========================================================================
    print("Step 1: Generating synthetic training data...")
    print("-" * 80)
    
    all_tickets = []
    tickets_per_category = 150  # 150 tickets per category for training
    
    for category in TICKET_CATEGORIES.keys():
        tickets = generate_synthetic_tickets(tickets_per_category, category)
        all_tickets.extend(tickets)
        print(f"  Generated {len(tickets)} tickets for {category}")
    
    print(f"\nTotal training tickets: {len(all_tickets)}")
    
    # Generate knowledge base
    print("\nGenerating knowledge base articles...")
    kb_articles = generate_knowledge_base()
    print(f"  Generated {len(kb_articles)} knowledge base articles")
    
    # Split into train and test
    random.shuffle(all_tickets)
    split_idx = int(len(all_tickets) * 0.8)
    train_tickets = all_tickets[:split_idx]
    test_tickets = all_tickets[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training tickets: {len(train_tickets)}")
    print(f"  Test tickets: {len(test_tickets)}")
    print()
    
    # ========================================================================
    # Step 2: Initialize and Train System
    # ========================================================================
    print("Step 2: Training the Technical Support System...")
    print("-" * 80)
    
    system = TechnicalSupportSystem(
        embedding_model='all-MiniLM-L6-v2',
        classifier_type='gradient_boosting',
        use_tfidf=True
    )
    
    training_report = system.train(
        training_tickets=train_tickets,
        knowledge_base=kb_articles,
        validation_split=0.2
    )
    
    print("\nTraining Report:")
    print(f"  Validation Accuracy: {training_report['classifier_metrics']['validation_accuracy']:.4f}")
    print(f"  CV Accuracy: {training_report['classifier_metrics']['cv_accuracy_mean']:.4f} "
          f"(+/- {training_report['classifier_metrics']['cv_accuracy_std']:.4f})")
    print(f"  Number of Categories: {training_report['classifier_metrics']['num_categories']}")
    print()
    
    # ========================================================================
    # Step 3: Evaluate System Performance
    # ========================================================================
    print("Step 3: Evaluating system performance...")
    print("-" * 80)
    
    evaluator = PerformanceEvaluator()
    
    # Evaluate classifier
    print("\nClassifier Evaluation:")
    classifier_metrics = evaluator.evaluate_classifier(system.classifier, test_tickets)
    
    print(f"  Accuracy: {classifier_metrics['accuracy']:.4f}")
    print(f"  Weighted Precision: {classifier_metrics['precision']:.4f}")
    print(f"  Weighted Recall: {classifier_metrics['recall']:.4f}")
    print(f"  Weighted F1-Score: {classifier_metrics['f1_score']:.4f}")
    
    # Print per-category performance
    print("\n  Per-Category Performance:")
    report = classifier_metrics['classification_report']
    for category in TICKET_CATEGORIES.keys():
        if category in report:
            cat_metrics = report[category]
            print(f"    {category:12s}: Precision={cat_metrics['precision']:.3f}, "
                  f"Recall={cat_metrics['recall']:.3f}, F1={cat_metrics['f1-score']:.3f}")
    
    # Evaluate recommender
    print("\nRecommender Evaluation:")
    ground_truth = generate_ground_truth_mapping(test_tickets, kb_articles)
    recommender_metrics = evaluator.evaluate_recommender(
        system.recommender,
        test_tickets,
        ground_truth,
        top_k=5
    )
    
    print(f"  Precision@5: {recommender_metrics['precision@5']:.4f}")
    print(f"  Recall@5: {recommender_metrics['recall@5']:.4f}")
    print(f"  NDCG@5: {recommender_metrics['ndcg@5']:.4f}")
    print(f"  MRR: {recommender_metrics['mrr']:.4f}")
    
    # Simulate resolution time improvement
    print("\nResolution Time Impact:")
    # Simulate improved resolution times (20-30% reduction)
    improved_tickets = []
    for ticket in test_tickets:
        improved_ticket = TicketData(
            ticket_id=ticket.ticket_id,
            title=ticket.title,
            description=ticket.description,
            category=ticket.category,
            priority=ticket.priority,
            resolution_time=ticket.resolution_time * random.uniform(0.7, 0.8),  # 20-30% faster
            resolution_notes=ticket.resolution_notes
        )
        improved_tickets.append(improved_ticket)
    
    resolution_metrics = evaluator.evaluate_resolution_time_impact(
        test_tickets,
        improved_tickets
    )
    
    print(f"  Baseline Mean: {resolution_metrics['baseline_mean_hours']:.2f} hours")
    print(f"  Improved Mean: {resolution_metrics['improved_mean_hours']:.2f} hours")
    print(f"  Reduction: {resolution_metrics['reduction_percentage']:.2f}%")
    print()
    
    # ========================================================================
    # Step 4: Demonstrate Real-time Processing
    # ========================================================================
    print("Step 4: Real-time ticket processing demonstration...")
    print("-" * 80)
    
    # Create sample tickets for demonstration
    demo_tickets = [
        TicketData(
            ticket_id="DEMO-001",
            title="VPN connection keeps dropping",
            description="Unable to maintain stable VPN connection. Connection drops every "
                       "few minutes. Getting timeout errors. This is urgent as I cannot "
                       "access company resources.",
            category="Network",  # Ground truth for demo
            priority="High",
            resolution_time=0.0
        ),
        TicketData(
            ticket_id="DEMO-002",
            title="Database query running extremely slow",
            description="Production database queries timing out. Performance degradation "
                       "started this morning. Affecting all users. Need immediate help "
                       "to identify bottleneck.",
            category="Database",
            priority="High",
            resolution_time=0.0
        ),
        TicketData(
            ticket_id="DEMO-003",
            title="Cannot login to account",
            description="Forgot password and reset link not working. Account locked after "
                       "multiple attempts. Need access urgently.",
            category="Account",
            priority="Medium",
            resolution_time=0.0
        )
    ]
    
    print("\nProcessing sample tickets:\n")
    
    for ticket in demo_tickets:
        print(f"Ticket ID: {ticket.ticket_id}")
        print(f"Title: {ticket.title}")
        print(f"Description: {ticket.description[:100]}...")
        print(f"Ground Truth Category: {ticket.category}")
        print()
        
        result = system.process_ticket(ticket, recommend_articles=True, top_k=3)
        
        print(f"Predicted Category: {result['predicted_category']} "
              f"(Confidence: {result['confidence']:.2%})")
        
        print("\nTop 3 Category Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['category']:12s} - {pred['confidence']:.2%}")
        
        print("\nRecommended Knowledge Base Articles:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. [{rec['article_id']}] {rec['title']}")
            print(f"     Category: {rec['category']}, Similarity: {rec['similarity_score']:.4f}")
            print(f"     Tags: {', '.join(rec['tags'])}")
        
        print("\n" + "=" * 80 + "\n")
    
    # ========================================================================
    # Step 5: Save the Trained System
    # ========================================================================
    print("Step 5: Saving trained system...")
    print("-" * 80)
    
    system.save('./trained_model')
    print("System saved to './trained_model/'")
    print()
    
    # ========================================================================
    # Step 6: Generate Performance Summary
    # ========================================================================
    print("Step 6: Performance Summary")
    print("-" * 80)
    
    summary_df = pd.DataFrame({
        'Metric': [
            'Classification Accuracy',
            'Weighted F1-Score',
            'Recommendation Precision@5',
            'Recommendation NDCG@5',
            'Resolution Time Reduction',
            'Training Tickets',
            'Test Tickets',
            'KB Articles'
        ],
        'Value': [
            f"{classifier_metrics['accuracy']:.2%}",
            f"{classifier_metrics['f1_score']:.2%}",
            f"{recommender_metrics['precision@5']:.2%}",
            f"{recommender_metrics['ndcg@5']:.4f}",
            f"{resolution_metrics['reduction_percentage']:.1f}%",
            len(train_tickets),
            len(test_tickets),
            len(kb_articles)
        ]
    })
    
    print(summary_df.to_string(index=False))
    print()
    
    print("=" * 80)
    print("Demonstration completed successfully!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Load the saved model: system = TechnicalSupportSystem.load('./trained_model')")
    print("  2. Process new tickets: result = system.process_ticket(new_ticket)")
    print("  3. Monitor performance and retrain as needed")
    print("  4. Deploy to production environment")
    print()


if __name__ == "__main__":
    main()

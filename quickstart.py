#!/usr/bin/env python3
"""
Quick Start Script for Technical Support Ticket Classification System

This script provides a guided walkthrough for setting up and running
the ticket classification system.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def print_step(number, title):
    """Print step header"""
    print(f"\n{'='*80}")
    print(f"STEP {number}: {title}")
    print(f"{'='*80}\n")


def check_dependencies():
    """Check if required dependencies are installed"""
    print_step(1, "Checking Dependencies")
    
    required = [
        'numpy',
        'pandas',
        'sklearn',
        'sentence_transformers',
        'flask',
        'torch'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def run_demo():
    """Run the demonstration script"""
    print_step(2, "Running Demo")
    
    print("This will:")
    print("  1. Generate synthetic training data")
    print("  2. Train the classification model")
    print("  3. Evaluate performance")
    print("  4. Demonstrate real-time processing")
    print("  5. Save the trained model")
    print()
    
    response = input("Continue with demo? (y/n): ").lower()
    if response != 'y':
        print("Skipping demo")
        return False
    
    print("\nRunning demo.py...")
    print("-" * 80)
    
    try:
        import demo
        demo.main()
        return True
    except Exception as e:
        print(f"Error running demo: {e}")
        return False


def test_system():
    """Run test suite"""
    print_step(3, "Running Tests")
    
    print("This will run the test suite to verify system functionality.")
    print()
    
    response = input("Run tests? (y/n): ").lower()
    if response != 'y':
        print("Skipping tests")
        return
    
    print("\nRunning tests...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'test_system.py', '-v'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed")
            print(result.stderr)
    except Exception as e:
        print(f"Error running tests: {e}")


def start_api_server():
    """Start the API server"""
    print_step(4, "Starting API Server")
    
    print("This will start the REST API server for production use.")
    print()
    print("API Endpoints:")
    print("  POST /api/v1/classify      - Classify a ticket")
    print("  POST /api/v1/recommend     - Get recommendations")
    print("  POST /api/v1/process       - Full processing")
    print("  GET  /api/v1/categories    - List categories")
    print("  GET  /health               - Health check")
    print()
    
    response = input("Start API server? (y/n): ").lower()
    if response != 'y':
        print("Skipping API server")
        return
    
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("-" * 80)
    
    try:
        import api_server
        api_server.init_system('./trained_model')
        api_server.app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")


def show_usage_examples():
    """Show usage examples"""
    print_step(5, "Usage Examples")
    
    examples = """
Example 1: Load and Use Trained Model
--------------------------------------
from ticket_classifier import TechnicalSupportSystem, TicketData

# Load model
system = TechnicalSupportSystem.load('./trained_model')

# Create a ticket
ticket = TicketData(
    ticket_id="T-12345",
    title="VPN connection issue",
    description="Cannot connect to VPN, getting timeout errors",
    category="Unknown",
    priority="High",
    resolution_time=0.0
)

# Process ticket
result = system.process_ticket(ticket, recommend_articles=True, top_k=5)

print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\\nRecommendations:")
for rec in result['recommendations']:
    print(f"  - {rec['title']}")


Example 2: API Request (using curl)
------------------------------------
# Classify a ticket
curl -X POST http://localhost:5000/api/v1/classify \\
  -H "Content-Type: application/json" \\
  -d '{
    "ticket_id": "T-001",
    "title": "Database slow",
    "description": "Queries timing out frequently"
  }'


Example 3: Batch Processing
----------------------------
from ticket_classifier import TechnicalSupportSystem

system = TechnicalSupportSystem.load('./trained_model')

tickets = [...]  # List of TicketData objects

for ticket in tickets:
    result = system.process_ticket(ticket)
    print(f"{ticket.ticket_id}: {result['predicted_category']}")


Example 4: Integration with JIRA
---------------------------------
from integrations import JiraIntegration
from ticket_classifier import TechnicalSupportSystem

system = TechnicalSupportSystem.load('./trained_model')

jira = JiraIntegration(
    jira_url='https://your-domain.atlassian.net',
    username='your-email@example.com',
    api_token='your-api-token',
    support_system=system
)

# Process all unclassified tickets
results = jira.batch_process_tickets(project_key='SUPPORT')


Example 5: Performance Benchmarking
------------------------------------
from benchmark import run_comprehensive_benchmark
from ticket_classifier import TechnicalSupportSystem

system = TechnicalSupportSystem.load('./trained_model')
run_comprehensive_benchmark(system)
"""
    
    print(examples)
    
    response = input("\nSave examples to file? (y/n): ").lower()
    if response == 'y':
        with open('usage_examples.txt', 'w') as f:
            f.write(examples)
        print("Examples saved to: usage_examples.txt")


def show_configuration():
    """Show configuration options"""
    print_step(6, "Configuration")
    
    print("Key configuration options in config.py:")
    print()
    print("Model Settings:")
    print("  EMBEDDING_MODEL = 'all-MiniLM-L6-v2'")
    print("  CLASSIFIER_TYPE = 'gradient_boosting'")
    print("  USE_TFIDF = True")
    print()
    print("Performance Settings:")
    print("  EMBEDDING_BATCH_SIZE = 32")
    print("  N_WORKERS = -1  # Use all CPU cores")
    print()
    print("API Settings:")
    print("  API_HOST = '0.0.0.0'")
    print("  API_PORT = 5000")
    print()
    print("Edit config.py to customize these settings.")


def main_menu():
    """Show main menu"""
    print_header("Technical Support Ticket Classification System")
    print("Quick Start Guide")
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install dependencies before continuing.")
        print("Run: pip install -r requirements.txt")
        return
    
    while True:
        print("\n" + "=" * 80)
        print("Main Menu")
        print("=" * 80)
        print()
        print("1. Run Demo (Train and test the system)")
        print("2. Run Tests (Verify functionality)")
        print("3. Start API Server (For production use)")
        print("4. Show Usage Examples")
        print("5. Show Configuration Options")
        print("6. Exit")
        print()
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            test_system()
        elif choice == '3':
            start_api_server()
        elif choice == '4':
            show_usage_examples()
        elif choice == '5':
            show_configuration()
        elif choice == '6':
            print("\nExiting...")
            break
        else:
            print("Invalid option. Please select 1-6.")


def quick_start():
    """Run quick start sequence"""
    print_header("Quick Start: Automated Setup")
    
    print("This will automatically:")
    print("  1. Check dependencies")
    print("  2. Run demo and train model")
    print("  3. Run tests")
    print()
    
    response = input("Continue? (y/n): ").lower()
    if response != 'y':
        main_menu()
        return
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return
    
    # Step 2: Run demo
    if not run_demo():
        print("\nDemo failed. Please check error messages.")
        return
    
    # Step 3: Run tests
    test_system()
    
    print_header("Quick Start Complete!")
    print("Next steps:")
    print("  1. Start API server: python api_server.py")
    print("  2. View usage examples: see usage_examples.txt")
    print("  3. Customize configuration: edit config.py")
    print()
    
    main_menu()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Technical Support System Quick Start'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run automated quick start'
    )
    
    args = parser.parse_args()
    
    if args.quick:
        quick_start()
    else:
        main_menu()

"""
Example Integration Scripts for Technical Support System

This module demonstrates how to integrate the ticket classification system
with popular ticketing platforms like JIRA, ServiceNow, Zendesk, and others.
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests

from ticket_classifier import (
    TechnicalSupportSystem,
    TicketData,
    KnowledgeBaseArticle
)


# ============================================================================
# JIRA Integration
# ============================================================================

class JiraIntegration:
    """Integration with Atlassian JIRA"""
    
    def __init__(
        self,
        jira_url: str,
        username: str,
        api_token: str,
        support_system: TechnicalSupportSystem
    ):
        self.jira_url = jira_url.rstrip('/')
        self.auth = (username, api_token)
        self.support_system = support_system
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def fetch_unclassified_tickets(self, project_key: str) -> List[Dict]:
        """Fetch tickets that need classification"""
        jql = f'project = {project_key} AND labels = "needs-classification"'
        
        response = requests.get(
            f'{self.jira_url}/rest/api/3/search',
            auth=self.auth,
            headers=self.headers,
            params={'jql': jql, 'maxResults': 100}
        )
        
        if response.status_code == 200:
            return response.json()['issues']
        else:
            raise Exception(f"Failed to fetch tickets: {response.text}")
    
    def process_ticket(self, jira_issue: Dict) -> Dict:
        """Process a JIRA ticket"""
        # Convert JIRA issue to TicketData
        ticket = TicketData(
            ticket_id=jira_issue['key'],
            title=jira_issue['fields']['summary'],
            description=jira_issue['fields'].get('description', ''),
            category='Unknown',
            priority=jira_issue['fields']['priority']['name'],
            resolution_time=0.0
        )
        
        # Process with support system
        result = self.support_system.process_ticket(ticket, recommend_articles=True)
        
        # Update JIRA ticket
        self.update_ticket_category(
            jira_issue['key'],
            result['predicted_category'],
            result['confidence']
        )
        
        # Add comment with recommendations
        self.add_recommendations_comment(
            jira_issue['key'],
            result['recommendations']
        )
        
        return result
    
    def update_ticket_category(
        self,
        ticket_key: str,
        category: str,
        confidence: float
    ) -> None:
        """Update JIRA ticket with predicted category"""
        # Assuming you have a custom field for category
        update_data = {
            'fields': {
                'customfield_10001': category,  # Replace with your field ID
                'labels': ['auto-classified']
            }
        }
        
        response = requests.put(
            f'{self.jira_url}/rest/api/3/issue/{ticket_key}',
            auth=self.auth,
            headers=self.headers,
            json=update_data
        )
        
        if response.status_code != 204:
            print(f"Warning: Failed to update category: {response.text}")
    
    def add_recommendations_comment(
        self,
        ticket_key: str,
        recommendations: List[Dict]
    ) -> None:
        """Add knowledge base recommendations as a comment"""
        if not recommendations:
            return
        
        comment_text = "🤖 *Recommended Knowledge Base Articles:*\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            comment_text += (
                f"{i}. *{rec['title']}* (ID: {rec['article_id']})\n"
                f"   Category: {rec['category']}, "
                f"Relevance: {rec['similarity_score']:.2%}\n"
                f"   Tags: {', '.join(rec['tags'])}\n\n"
            )
        
        comment_data = {
            'body': comment_text
        }
        
        response = requests.post(
            f'{self.jira_url}/rest/api/3/issue/{ticket_key}/comment',
            auth=self.auth,
            headers=self.headers,
            json=comment_data
        )
        
        if response.status_code != 201:
            print(f"Warning: Failed to add comment: {response.text}")
    
    def batch_process_tickets(self, project_key: str) -> List[Dict]:
        """Process all unclassified tickets in a project"""
        tickets = self.fetch_unclassified_tickets(project_key)
        results = []
        
        for jira_issue in tickets:
            try:
                result = self.process_ticket(jira_issue)
                results.append({
                    'ticket_key': jira_issue['key'],
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'ticket_key': jira_issue['key'],
                    'success': False,
                    'error': str(e)
                })
        
        return results


# ============================================================================
# ServiceNow Integration
# ============================================================================

class ServiceNowIntegration:
    """Integration with ServiceNow"""
    
    def __init__(
        self,
        instance_url: str,
        username: str,
        password: str,
        support_system: TechnicalSupportSystem
    ):
        self.instance_url = instance_url.rstrip('/')
        self.auth = (username, password)
        self.support_system = support_system
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def fetch_new_incidents(self, limit: int = 100) -> List[Dict]:
        """Fetch new incidents that need classification"""
        params = {
            'sysparm_query': 'state=1^category=NULL',  # State=1 is New
            'sysparm_limit': limit,
            'sysparm_fields': 'sys_id,number,short_description,description,priority'
        }
        
        response = requests.get(
            f'{self.instance_url}/api/now/table/incident',
            auth=self.auth,
            headers=self.headers,
            params=params
        )
        
        if response.status_code == 200:
            return response.json()['result']
        else:
            raise Exception(f"Failed to fetch incidents: {response.text}")
    
    def process_incident(self, incident: Dict) -> Dict:
        """Process a ServiceNow incident"""
        ticket = TicketData(
            ticket_id=incident['number'],
            title=incident['short_description'],
            description=incident.get('description', ''),
            category='Unknown',
            priority=incident.get('priority', '3'),
            resolution_time=0.0
        )
        
        result = self.support_system.process_ticket(ticket, recommend_articles=True)
        
        # Update incident with category
        self.update_incident(
            incident['sys_id'],
            result['predicted_category'],
            result['recommendations']
        )
        
        return result
    
    def update_incident(
        self,
        sys_id: str,
        category: str,
        recommendations: List[Dict]
    ) -> None:
        """Update ServiceNow incident with classification"""
        # Build work notes with recommendations
        work_notes = "Auto-classified. Recommended KB articles:\n"
        for rec in recommendations[:3]:
            work_notes += f"- {rec['title']} ({rec['article_id']})\n"
        
        update_data = {
            'category': category,
            'work_notes': work_notes,
            'assignment_group': self._get_assignment_group(category)
        }
        
        response = requests.patch(
            f'{self.instance_url}/api/now/table/incident/{sys_id}',
            auth=self.auth,
            headers=self.headers,
            json=update_data
        )
        
        if response.status_code != 200:
            print(f"Warning: Failed to update incident: {response.text}")
    
    def _get_assignment_group(self, category: str) -> str:
        """Map category to assignment group"""
        category_to_group = {
            'Network': 'network_support',
            'Hardware': 'hardware_support',
            'Software': 'application_support',
            'Database': 'database_team',
            'Security': 'security_team',
            'Email': 'email_support',
            'Cloud': 'cloud_ops',
            'Account': 'helpdesk'
        }
        return category_to_group.get(category, 'general_support')


# ============================================================================
# Zendesk Integration
# ============================================================================

class ZendeskIntegration:
    """Integration with Zendesk"""
    
    def __init__(
        self,
        subdomain: str,
        email: str,
        api_token: str,
        support_system: TechnicalSupportSystem
    ):
        self.base_url = f'https://{subdomain}.zendesk.com/api/v2'
        self.auth = (f'{email}/token', api_token)
        self.support_system = support_system
        self.headers = {'Content-Type': 'application/json'}
    
    def fetch_new_tickets(self, status: str = 'new') -> List[Dict]:
        """Fetch new tickets"""
        response = requests.get(
            f'{self.base_url}/search.json',
            auth=self.auth,
            headers=self.headers,
            params={'query': f'type:ticket status:{status}'}
        )
        
        if response.status_code == 200:
            return response.json()['results']
        else:
            raise Exception(f"Failed to fetch tickets: {response.text}")
    
    def process_ticket(self, zendesk_ticket: Dict) -> Dict:
        """Process a Zendesk ticket"""
        ticket = TicketData(
            ticket_id=str(zendesk_ticket['id']),
            title=zendesk_ticket['subject'],
            description=zendesk_ticket['description'],
            category='Unknown',
            priority=zendesk_ticket.get('priority', 'normal'),
            resolution_time=0.0
        )
        
        result = self.support_system.process_ticket(ticket, recommend_articles=True)
        
        # Update ticket with tags and internal note
        self.update_ticket(
            zendesk_ticket['id'],
            result['predicted_category'],
            result['recommendations']
        )
        
        return result
    
    def update_ticket(
        self,
        ticket_id: int,
        category: str,
        recommendations: List[Dict]
    ) -> None:
        """Update Zendesk ticket"""
        # Create internal note with recommendations
        note = "AI Classification Results:\n"
        note += f"Category: {category}\n\n"
        note += "Recommended Articles:\n"
        for rec in recommendations[:3]:
            note += f"- {rec['title']} (Relevance: {rec['similarity_score']:.1%})\n"
        
        update_data = {
            'ticket': {
                'tags': [category.lower(), 'auto-classified'],
                'comment': {
                    'body': note,
                    'public': False
                }
            }
        }
        
        response = requests.put(
            f'{self.base_url}/tickets/{ticket_id}.json',
            auth=self.auth,
            headers=self.headers,
            json=update_data
        )
        
        if response.status_code != 200:
            print(f"Warning: Failed to update ticket: {response.text}")


# ============================================================================
# Generic Webhook Integration
# ============================================================================

class WebhookIntegration:
    """Generic webhook-based integration"""
    
    def __init__(
        self,
        webhook_url: str,
        support_system: TechnicalSupportSystem,
        auth_token: Optional[str] = None
    ):
        self.webhook_url = webhook_url
        self.support_system = support_system
        self.headers = {'Content-Type': 'application/json'}
        
        if auth_token:
            self.headers['Authorization'] = f'Bearer {auth_token}'
    
    def process_webhook_payload(self, payload: Dict) -> Dict:
        """Process incoming webhook payload"""
        # Extract ticket information from payload
        ticket = TicketData(
            ticket_id=payload.get('ticket_id', 'UNKNOWN'),
            title=payload.get('title', ''),
            description=payload.get('description', ''),
            category='Unknown',
            priority=payload.get('priority', 'Medium'),
            resolution_time=0.0
        )
        
        # Process ticket
        result = self.support_system.process_ticket(ticket, recommend_articles=True)
        
        # Send result back via webhook
        self.send_result(result)
        
        return result
    
    def send_result(self, result: Dict) -> None:
        """Send processing result via webhook"""
        response = requests.post(
            self.webhook_url,
            headers=self.headers,
            json=result
        )
        
        if response.status_code not in [200, 201, 204]:
            print(f"Warning: Failed to send result: {response.text}")


# ============================================================================
# Email Integration
# ============================================================================

class EmailIntegration:
    """Process tickets from email"""
    
    def __init__(
        self,
        imap_server: str,
        email_address: str,
        password: str,
        support_system: TechnicalSupportSystem
    ):
        import imaplib
        import email
        from email.header import decode_header
        
        self.imap = imaplib.IMAP4_SSL(imap_server)
        self.imap.login(email_address, password)
        self.support_system = support_system
        self.email_module = email
        self.decode_header = decode_header
    
    def process_unread_emails(self, mailbox: str = 'INBOX') -> List[Dict]:
        """Process unread emails as tickets"""
        self.imap.select(mailbox)
        
        # Search for unread emails
        _, message_numbers = self.imap.search(None, 'UNSEEN')
        results = []
        
        for num in message_numbers[0].split():
            _, msg_data = self.imap.fetch(num, '(RFC822)')
            email_body = msg_data[0][1]
            email_message = self.email_module.message_from_bytes(email_body)
            
            # Extract email details
            subject = self._decode_subject(email_message['subject'])
            body = self._get_email_body(email_message)
            from_address = email_message['from']
            
            # Create ticket
            ticket = TicketData(
                ticket_id=f"EMAIL-{num.decode()}",
                title=subject,
                description=body,
                category='Unknown',
                priority='Medium',
                resolution_time=0.0
            )
            
            # Process
            result = self.support_system.process_ticket(ticket, recommend_articles=True)
            result['from'] = from_address
            
            results.append(result)
        
        return results
    
    def _decode_subject(self, subject):
        """Decode email subject"""
        decoded = self.decode_header(subject)[0]
        if isinstance(decoded[0], bytes):
            return decoded[0].decode(decoded[1] or 'utf-8')
        return decoded[0]
    
    def _get_email_body(self, email_message):
        """Extract email body"""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_payload(decode=True).decode()
        else:
            return email_message.get_payload(decode=True).decode()


# ============================================================================
# Example Usage
# ============================================================================

def example_jira_integration():
    """Example: Integrate with JIRA"""
    
    # Load trained system
    system = TechnicalSupportSystem.load('./trained_model')
    
    # Initialize JIRA integration
    jira = JiraIntegration(
        jira_url='https://your-domain.atlassian.net',
        username='your-email@example.com',
        api_token='your-api-token',
        support_system=system
    )
    
    # Process all unclassified tickets
    results = jira.batch_process_tickets(project_key='SUPPORT')
    
    print(f"Processed {len(results)} tickets")
    for result in results:
        if result['success']:
            print(f"  ✓ {result['ticket_key']}: {result['result']['predicted_category']}")
        else:
            print(f"  ✗ {result['ticket_key']}: {result['error']}")


def example_servicenow_integration():
    """Example: Integrate with ServiceNow"""
    
    system = TechnicalSupportSystem.load('./trained_model')
    
    snow = ServiceNowIntegration(
        instance_url='https://your-instance.service-now.com',
        username='admin',
        password='password',
        support_system=system
    )
    
    # Fetch and process new incidents
    incidents = snow.fetch_new_incidents(limit=50)
    
    for incident in incidents:
        result = snow.process_incident(incident)
        print(f"Processed {incident['number']}: {result['predicted_category']}")


def example_webhook_handler():
    """Example: Webhook handler for real-time processing"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    system = TechnicalSupportSystem.load('./trained_model')
    webhook = WebhookIntegration('https://callback.url', system)
    
    @app.route('/webhook/ticket', methods=['POST'])
    def handle_ticket():
        payload = request.get_json()
        result = webhook.process_webhook_payload(payload)
        return jsonify(result)
    
    app.run(port=8080)


if __name__ == '__main__':
    print("Integration Examples")
    print("=" * 80)
    print()
    print("1. JIRA Integration - example_jira_integration()")
    print("2. ServiceNow Integration - example_servicenow_integration()")
    print("3. Webhook Handler - example_webhook_handler()")
    print()
    print("Modify the examples with your actual credentials and run.")

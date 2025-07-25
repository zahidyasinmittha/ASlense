# app/services/email_service.py
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class EmailService:
    """Service for handling email operations"""
    
    def __init__(self):
        self.smtp_server = settings.EMAIL_HOST
        self.smtp_port = settings.EMAIL_PORT
        self.username = settings.EMAIL_HOST_USER
        self.password = settings.EMAIL_HOST_PASSWORD
        self.use_tls = settings.EMAIL_USE_TLS
        self.enabled = settings.EMAIL_SEND

    async def send_email(
        self,
        to_emails: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        from_email: Optional[str] = None
    ) -> bool:
        """Send an email"""
        if not self.enabled:
            logger.info("Email sending is disabled")
            return True
            
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email or self.username
            msg['To'] = ', '.join(to_emails)
            
            # Add plain text part
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {to_emails}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    async def send_contact_form_email(
        self,
        name: str,
        email: str,
        subject: str,
        message: str
    ) -> bool:
        """Send contact form email to admin"""
        email_subject = f"ASLense Contact Form: {subject}"
        
        # Plain text body
        text_body = f"""
New contact form submission from ASLense website:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
Sent at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
From: ASLense Contact Form
        """.strip()
        
        # HTML body
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background-color: #4F46E5; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; }}
        .field {{ margin-bottom: 15px; }}
        .label {{ font-weight: bold; color: #4F46E5; }}
        .value {{ margin-top: 5px; }}
        .message {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4F46E5; }}
        .footer {{ background-color: #f8f9fa; padding: 10px; text-align: center; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🤟 ASLense Contact Form Submission</h2>
    </div>
    
    <div class="content">
        <p>You have received a new contact form submission from the ASLense website:</p>
        
        <div class="field">
            <div class="label">Name:</div>
            <div class="value">{name}</div>
        </div>
        
        <div class="field">
            <div class="label">Email:</div>
            <div class="value">{email}</div>
        </div>
        
        <div class="field">
            <div class="label">Subject:</div>
            <div class="value">{subject}</div>
        </div>
        
        <div class="field">
            <div class="label">Message:</div>
            <div class="message">{message.replace(chr(10), '<br>')}</div>
        </div>
    </div>
    
    <div class="footer">
        <p>Sent at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} from ASLense Contact Form</p>
    </div>
</body>
</html>
        """
        
        return await self.send_email(
            to_emails=[settings.CONTACT_RECIPIENT_EMAIL],
            subject=email_subject,
            body=text_body,
            html_body=html_body
        )

# Global email service instance
email_service = EmailService()

from app import db
from twilio.rest import Client
import os

# Twilio configuration
twilio_client = None
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

try:
    if os.getenv('TWILIO_ACCOUNT_SID') and os.getenv('TWILIO_AUTH_TOKEN'):
        twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
except Exception as e:
    print(f"Twilio initialization failed: {e}")

def create_notification(recipient_id, sender_id, food_id, notification_type, title, message, area):
    """Create a new notification in the database"""
    try:
        from models import Notification
        
        notification = Notification(
            recipient_id=recipient_id,
            sender_id=sender_id,
            food_id=food_id,
            type=notification_type,
            title=title,
            message=message,
            area=area
        )
        
        db.session.add(notification)
        db.session.commit()
        
        return notification
        
    except Exception as e:
        db.session.rollback()
        print(f"Error creating notification: {str(e)}")
        return None

def send_sms_notification(phone, message):
    """Send SMS notification using Twilio"""
    try:
        if not TWILIO_PHONE_NUMBER or not twilio_client:
            print("Twilio not configured, skipping SMS")
            return False
        
        # Format phone number for India (+91)
        if not phone.startswith('+'):
            phone = '+91' + phone
        
        message_instance = twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone
        )
        
        print(f"SMS sent successfully: {message_instance.sid}")
        return True
        
    except Exception as e:
        print(f"Error sending SMS: {str(e)}")
        return False

def send_bulk_sms_to_area(area, message, user_type='ngo'):
    """Send SMS to all users of a specific type in an area"""
    try:
        from models import User
        
        users = User.query.filter_by(area=area, user_type=user_type).all()
        
        success_count = 0
        for user in users:
            if send_sms_notification(user.phone, message):
                success_count += 1
        
        return success_count
        
    except Exception as e:
        print(f"Error sending bulk SMS: {str(e)}")
        return 0
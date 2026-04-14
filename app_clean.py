from flask import Flask, request, jsonify, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import qrcode
import io
import base64
import string
import random
import threading
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
import cv2
from PIL import Image as PILImage
import torch
import torchvision.transforms as T
from torchvision import models as tv_models

# ── Inline Food Quality Analyzer ──────────────────────────────────────────────
_fq_model = None
_fq_labels = None

def _get_mobilenet():
    global _fq_model, _fq_labels
    if _fq_model is None:
        _fq_model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
        _fq_model.eval()
        import urllib.request, json
        try:
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            with urllib.request.urlopen(url, timeout=5) as r:
                _fq_labels = json.loads(r.read().decode())
        except Exception:
            _fq_labels = [f"class_{i}" for i in range(1000)]
    return _fq_model, _fq_labels

_FOOD_WORDS = {
    'apple','banana','orange','lemon','pineapple','strawberry','pizza','hamburger',
    'hotdog','sandwich','soup','rice','bread','broccoli','carrot','corn','cucumber',
    'mushroom','onion','pepper','potato','tomato','egg','cheese','butter','yogurt',
    'meat','chicken','fish','salad','vegetable','fruit','mango','dal','roti',
    'fig','grape','pear','peach','guava','papaya','cauliflower','spinach','eggplant',
    'zucchini','lettuce','cabbage','pumpkin','radish','beetroot','drumstick',
    'noodle','pasta','dumpling','sushi','taco','burrito','waffle','pancake',
    'donut','cake','cookie','muffin','croissant','pretzel','bagel',
    'mashed','cooked','boiled','steamed','fried','roasted','grilled','baked',
    'white','grain','cereal','starch','staple','food','meal','dish','cuisine',
    # cooking vessels and utensils that contain/serve food → treat as food context
    'wok','bowl','pot','pan','plate','dish','tray','cup','jar','casserole',
    'skillet','saucepan','stew','curry','biryani','pulao','khichdi','sabzi',
    'chapati','paratha','idli','dosa','samosa','pakora','chutney','pickle',
    'pudding','custard','ice cream','gelato','sorbet','smoothie','juice',
    'coffee','tea','milk','lassi','buttermilk','porridge','oatmeal','cereal',
    # utensils used with food → food context
    'spoon','ladle','fork','chopstick','spatula','wooden spoon','serving',
    'scoop','tongs','whisk','grater','colander','strainer'
}
# Words that are NOT food - explicitly reject these
_NON_FOOD_WORDS = {
    'turtle','sea turtle','loggerhead',
    'person','man','woman','child','face','human','people',
    'car','vehicle','truck','bus','motorcycle','bicycle','airplane','helicopter',
    'warplane','tank','ship','boat','train','locomotive','steam locomotive',
    'military','soldier','uniform','weapon','gun','rifle',
    'building','house','temple','church','mosque','statue','sculpture',
    'painting','mask','costume','pencil',
    'dog','cat','bird','horse','elephant','tiger','lion','monkey',
    'snake','lizard','insect','spider','jellyfish',
    'computer','phone','keyboard','screen','book','paper','pen',
    'flower','tree','grass','rock','mountain','sky','cloud'
}
_SPOIL_WORDS = {'mold','fungus','rot','decay','compost','waste','garbage','trash'}

def _analyze_food_image_inline(image_bytes):
    print(f"[FQ] Analyzing {len(image_bytes)} bytes")

    # ── Try Gemini Vision API first ───────────────────────────────────────────
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini_key)

            pil_img = PILImage.open(io.BytesIO(image_bytes)).convert('RGB')
            # Convert PIL to bytes for Gemini
            img_buf = io.BytesIO()
            pil_img.save(img_buf, format='JPEG')
            img_bytes = img_buf.getvalue()

            prompt = """Analyze this image for food donation safety. Reply ONLY with JSON, no extra text:
{
  "is_food": true or false,
  "food_name": "name of food or what you see",
  "quality": "fresh" or "moderate" or "poor" or "not_food",
  "confidence": number 0-100,
  "reason": "one line reason"
}
Rules:
- ANY edible item (rice, dal, roti, vegetables, fruits, cooked food, raw food, packaged food) = is_food TRUE
- Mashed potato, cooked rice, dal, curry, chapati, any Indian food = is_food TRUE
- NOT food ONLY IF: person, vehicle, animal (non-food), building, landscape, text document
- Food WITH utensils (spoon, fork, bowl, plate, pan) = is_food TRUE
- Fresh cooked food = quality fresh, confidence 80-95
- Slightly old but edible = quality moderate
- Moldy, rotten, clearly spoiled = quality poor
- Non-food object = quality not_food"""

            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    prompt
                ]
            )
            text = response.text.strip()
            print(f"[Gemini] Response: {text}")

            import json, re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_food = result.get('is_food', False)
                quality = result.get('quality', 'poor')
                confidence = int(result.get('confidence', 70))
                food_name = result.get('food_name', 'food')
                reason = result.get('reason', '')

                if not is_food or quality == 'not_food':
                    return {"quality": "unclear", "confidence": 0,
                            "message": f"⚠️ This is not a food image ({food_name}). Please upload a food photo."}
                elif quality == 'fresh':
                    return {"quality": "fresh", "confidence": confidence,
                            "message": f"Food looks fresh and safe for donation ({food_name}). {reason}"}
                elif quality == 'moderate':
                    return {"quality": "moderate", "confidence": confidence,
                            "message": f"Food quality acceptable - inspect before distribution ({food_name}). {reason}"}
                else:
                    return {"quality": "poor", "confidence": confidence,
                            "message": f"Food not safe for donation. {reason}"}
        except Exception as e:
            print(f"[Gemini] Error: {e}, falling back to OpenCV")

    # ── Fallback: OpenCV + MobileNet ─────────────────────────────────────────
    try:
        pil = PILImage.open(io.BytesIO(image_bytes)).convert('RGB')
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        print(f"[FQ] Decoded OK: {img.shape}")
    except Exception as e:
        print(f"[FQ] Decode failed: {e}")
        return {"quality": "error", "confidence": 0, "message": "Invalid image - could not read file"}

    img = cv2.resize(img, (320, 320))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = 320 * 320
    food_mask = cv2.inRange(hsv, np.array([0,0,30]), np.array([180,255,255]))
    food_px = int(np.sum(food_mask > 0))

    if food_px < total * 0.04:
        return {"quality": "fresh", "confidence": 65, "message": "Image unclear - assuming fresh"}

    sat = hsv[:,:,1][food_mask>0].astype(float)
    val = hsv[:,:,2][food_mask>0].astype(float)
    hue = hsv[:,:,0][food_mask>0].astype(float)
    mean_sat = float(np.mean(sat))
    mean_val = float(np.mean(val))
    hue_std  = float(np.std(hue))

    wm  = cv2.inRange(hsv, np.array([0,0,180]),  np.array([180,15,255]))
    gm  = cv2.inRange(hsv, np.array([0,0,80]),   np.array([180,30,180]))
    grm = cv2.inRange(hsv, np.array([35,30,20]),  np.array([85,180,140]))
    herb= cv2.inRange(hsv, np.array([35,100,140]),np.array([85,255,255]))
    white_r = float(np.sum(cv2.bitwise_and(wm,  food_mask)>0)) / food_px
    grey_r  = float(np.sum(cv2.bitwise_and(gm,  food_mask)>0)) / food_px
    green_r = max(0, int(np.sum(cv2.bitwise_and(grm,food_mask)>0))
                   - int(np.sum(cv2.bitwise_and(herb,food_mask)>0))) / food_px
    mold_r  = white_r + grey_r + green_r

    vib1 = cv2.inRange(hsv, np.array([0,80,80]),   np.array([35,255,255]))
    vib2 = cv2.inRange(hsv, np.array([155,80,60]), np.array([180,255,255]))
    vib_r = (int(np.sum(cv2.bitwise_and(vib1,food_mask)>0)) +
             int(np.sum(cv2.bitwise_and(vib2,food_mask)>0))) / food_px

    print(f"[FQ] sat={mean_sat:.1f} val={mean_val:.1f} hue_std={hue_std:.1f} mold={mold_r:.3f} vib={vib_r:.3f}")

    # ── MobileNet check FIRST ──────────────────────────────────────────────────
    top_label, is_food, is_spoiled, top5 = "unknown", False, False, []
    try:
        model, labels = _get_mobilenet()
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        tensor = tf(pil).unsqueeze(0)
        with torch.no_grad():
            out = torch.nn.functional.softmax(model(tensor)[0], dim=0)
        top5_p, top5_i = torch.topk(out, 5)
        top5 = [(labels[i], float(p)) for i,p in zip(top5_i, top5_p)]
        print(f"[FQ] Top5: {top5}")
        top_label = top5[0][0]

        # Check food and non-food words in top5 labels
        food_match     = any(any(w in l.lower() for w in _FOOD_WORDS)     for l,_ in top5)
        non_food_match = any(any(w in l.lower() for w in _NON_FOOD_WORDS) for l,_ in top5)

        if non_food_match and not food_match:
            # Non-food detected AND no food word → reject
            is_food = False
        else:
            # Food word found OR neither → assume food (safe default)
            is_food = True

        # Spoilage only relevant if it's actually food
        is_spoiled = is_food and any(any(w in l.lower() for w in _SPOIL_WORDS) for l,_ in top5)

    except Exception as e:
        print(f"[FQ] MobileNet error: {e}")
        is_food = True  # fallback: assume food if model fails

    # ── Non-food image → reject immediately ───────────────────────────────────
    if not is_food:
        return {"quality": "unclear", "confidence": 0,
                "message": f"⚠️ This is not a food image ({top_label} detected). Please upload a food photo."}
    if is_spoiled:
        return {"quality": "poor", "confidence": 15,
                "message": f"Spoilage detected ({top_label}) - NOT safe for donation"}

    # ── Hard mold rejection (only for confirmed food images) ──────────────────
    # Note: white foods (rice, idli, bread) are naturally white - use very high thresholds
    if white_r > 0.60: return {"quality":"poor","confidence":8,"message":"White mold detected - NOT safe"}
    if grey_r  > 0.50: return {"quality":"poor","confidence":10,"message":"Grey mold detected - NOT safe"}
    if green_r > 0.25: return {"quality":"poor","confidence":10,"message":"Green mold detected - NOT safe"}
    if mold_r  > 0.70: return {"quality":"poor","confidence":8,"message":"Mold detected - NOT safe"}

    # ── Score (starts at 70 = biased SAFE) ────────────────────────────────────
    score = 70
    top5_conf = top5[0][1] if top5 else 0
    score += 20 if top5_conf > 0.3 else 10

    score += 15 if vib_r>0.30 else (8 if vib_r>0.15 else (4 if vib_r>0.05 else 0))
    score += 8  if mean_sat>90 else (4 if mean_sat>60 else 0)
    score += 6  if hue_std>40  else (3 if hue_std>20  else 0)
    score += 4  if mean_val>130 else (-6 if mean_val<55 else 0)
    score -= 20 if white_r>0.55 else 0
    score -= 15 if grey_r>0.45  else 0
    score -= 15 if green_r>0.20 else 0

    score = max(0, min(100, score))
    print(f"[FQ] Final score={score} label={top_label} is_food={is_food} white_r={white_r:.3f}")

    if score >= 65:
        return {"quality":"fresh",    "confidence":score, "message":f"Food looks fresh and safe ({top_label} detected)"}
    elif score >= 40:
        return {"quality":"moderate", "confidence":score, "message":f"Acceptable quality - inspect before distribution ({top_label})"}
    else:
        return {"quality":"poor",     "confidence":score, "message":"Food shows signs of spoilage"}
# ── End Inline Food Quality Analyzer ──────────────────────────────────────────


load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///solapur_food_share.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-string')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Configure JWT to use user ID as identity
@jwt.user_identity_loader
def user_identity_lookup(user):
    return user

@jwt.user_lookup_loader
def user_lookup_callback(_jwt_header, jwt_data):
    identity = jwt_data["sub"]
    return User.query.filter_by(id=identity).one_or_none()

cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Twilio configuration (optional)
try:
    from twilio.rest import Client
    if os.getenv('TWILIO_ACCOUNT_SID') and os.getenv('TWILIO_AUTH_TOKEN'):
        twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    else:
        twilio_client = None
        TWILIO_PHONE_NUMBER = None
except ImportError:
    twilio_client = None
    TWILIO_PHONE_NUMBER = None

# Email Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() in ['true', 'on', '1']
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', '')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', '')

# ── NGO SMS Notification (no login needed) ───────────────────────────────────
def send_sms_to_ngo_contacts(food):
    """Send SMS to all NGO contacts in same taluka/area using ngo_contacts table"""
    try:
        if not twilio_client or not TWILIO_PHONE_NUMBER:
            print("[SMS] Twilio not configured — skipping SMS")
            return

        # Find matching NGO contacts by taluka or area
        from sqlalchemy import text
        query = """
            SELECT name, phone FROM ngo_contacts
            WHERE active = 1
            AND (taluka = :taluka OR area = :area OR taluka = :area OR area = :taluka)
        """
        with app.app_context():
            result = db.session.execute(text(query), {
                'taluka': food.taluka or '',
                'area':   food.area or ''
            }).fetchall()

        if not result:
            print(f"[SMS] No NGO contacts found for taluka={food.taluka} area={food.area}")
            return

        expiry_str = food.expiry_time.strftime('%d %b %I:%M %p') if food.expiry_time else 'N/A'
        message = (
            f"🍽️ New Food Available!\n"
            f"Food: {food.title}\n"
            f"Qty: {food.quantity}\n"
            f"Location: {food.area or food.taluka}\n"
            f"Pickup: {food.pickup_address}\n"
            f"Expires: {expiry_str}\n"
            f"Donor: {food.contact_person_name or 'N/A'}\n"
            f"Contact No: {food.contact_person_phone or 'N/A'}"
        )

        for name, phone in result:
            try:
                # Normalize phone number
                phone_clean = phone.strip().replace(' ', '').replace('-', '')
                if not phone_clean.startswith('+'):
                    if phone_clean.startswith('0'):
                        phone_clean = '+91' + phone_clean[1:]
                    elif phone_clean.startswith('91'):
                        phone_clean = '+' + phone_clean
                    else:
                        phone_clean = '+91' + phone_clean

                twilio_client.messages.create(
                    body=message,
                    from_=TWILIO_PHONE_NUMBER,
                    to=phone_clean
                )
                print(f"[SMS] ✅ Sent to {name} ({phone_clean})")
            except Exception as e:
                print(f"[SMS] ❌ Failed for {name} ({phone}): {e}")

    except Exception as e:
        print(f"[SMS] Error in send_sms_to_ngo_contacts: {e}")

# Email sending function
def send_email_notification(to_email, subject, body, food_id):
    """Send email notification and log the result"""
    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add body to email
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email using SMTP
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)
        
        # Log successful email
        email_log = EmailLog(
            ngo_email=to_email,
            food_id=food_id,
            status='sent'
        )
        db.session.add(email_log)
        db.session.commit()
        
        print(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        # Log failed email
        email_log = EmailLog(
            ngo_email=to_email,
            food_id=food_id,
            status='failed',
            error_message=str(e)
        )
        db.session.add(email_log)
        db.session.commit()
        
        print(f"Failed to send email to {to_email}: {str(e)}")
        return False

def send_food_donation_emails(food):
    """Send email notifications to all NGOs in the same taluka or area"""
    try:
        matched_ngos = set()

        # Match 1: NGO taluka == food taluka
        if food.taluka:
            for ngo in User.query.filter_by(user_type='ngo', taluka=food.taluka).all():
                matched_ngos.add(ngo.id)

        # Match 2: NGO area == food taluka (NGO has no taluka set)
        if food.taluka:
            for ngo in User.query.filter(
                User.user_type == 'ngo',
                User.taluka == None,
                User.area == food.taluka
            ).all():
                matched_ngos.add(ngo.id)

        # Match 3: NGO area == food area
        if food.area:
            for ngo in User.query.filter_by(user_type='ngo', area=food.area).all():
                matched_ngos.add(ngo.id)

        ngos = User.query.filter(User.id.in_(matched_ngos)).all() if matched_ngos else []

        if not ngos:
            print(f"No NGOs found in taluka: {food.taluka} / area: {food.area}")
            return
        
        # Prepare email content
        subject = f"New Food Donation Available in {food.taluka or food.area}"

        base_url = os.getenv('BASE_URL', 'http://localhost:5000')

        body = f"""Hello NGO Team,

A new food donation is available in your taluka. Please act quickly before it expires!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOOD DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Food Name     : {food.title}
Quantity      : {food.quantity}
Food Type     : {food.food_type or 'Not specified'}
Container     : {getattr(food, 'container_type', 'Not specified') or 'Not specified'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Area          : {food.area or 'N/A'}
Taluka        : {food.taluka or 'N/A'}
Pickup Address: {food.pickup_address}
Contact       : {food.contact_person_name or 'N/A'} - {food.contact_person_phone or 'N/A'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expiry Time   : {food.expiry_time.strftime('%d %b %Y, %I:%M %p')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please login to the platform and claim this food:
{base_url}/login

Thank you for your support in reducing food waste.

- Solapur Food Share Team
"""
        
        # Send email to each NGO
        for ngo in ngos:
            send_email_notification(ngo.email, subject, body, food.id)
        
        print(f"Email notifications sent to {len(ngos)} NGOs in {food.taluka}")
        
    except Exception as e:
        print(f"Error sending email notifications: {str(e)}")

# Solapur district areas
SOLAPUR_AREAS = [
    'Solapur City',
    'Pandharpur',
    'Akkalkot',
    'Barshi',
    'Karmala',
    'Madha',
    'Malshiras',
    'Mangalvedhe',
    'Mohol',
    'North Solapur',
    'South Solapur'
]

# Models
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    user_type = db.Column(db.Enum('donor', 'ngo', 'admin', name='user_types'), nullable=False)
    taluka = db.Column(db.String(50), nullable=True)
    area = db.Column(db.String(100), nullable=True)
    address = db.Column(db.Text, nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    
    # NGO specific fields
    organization_name = db.Column(db.String(200))
    registration_number = db.Column(db.String(50))
    capacity = db.Column(db.Integer)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'user_type': self.user_type,
            'taluka': self.taluka,
            'area': self.area,
            'address': self.address,
            'is_verified': self.is_verified,
            'organization_name': self.organization_name,
            'registration_number': self.registration_number,
            'capacity': self.capacity,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Food(db.Model):
    __tablename__ = 'foods'
    
    id = db.Column(db.Integer, primary_key=True)
    donor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    food_type = db.Column(db.Enum('cooked', 'raw', 'packaged', 'fruits', 'vegetables', 'dairy', 'other', name='food_types'), nullable=False)
    quantity = db.Column(db.String(100), nullable=False)
    expiry_time = db.Column(db.DateTime, nullable=False)
    pickup_address = db.Column(db.Text, nullable=False)
    taluka = db.Column(db.String(50))  # Taluka field
    area = db.Column(db.String(100))  # Changed from Enum to String
    images = db.Column(db.JSON)
    status = db.Column(db.Enum('available', 'claimed', 'picked_up', 'expired', name='food_status'), default='available')
    claimed_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    claimed_at = db.Column(db.DateTime)
    qr_code = db.Column(db.String(255), unique=True)
    pickup_code = db.Column(db.String(10), unique=True)
    contact_person_name = db.Column(db.String(100))
    contact_person_phone = db.Column(db.String(15))
    special_instructions = db.Column(db.Text)
    gps_latitude = db.Column(db.Float)
    gps_longitude = db.Column(db.Float)
    claimed_ngo_name = db.Column(db.String(200))
    claimed_ngo_phone = db.Column(db.String(20))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    donor = db.relationship('User', foreign_keys=[donor_id], backref='donated_foods')
    claimer = db.relationship('User', foreign_keys=[claimed_by], backref='claimed_foods')
    
    def __init__(self, **kwargs):
        super(Food, self).__init__(**kwargs)
        if not self.pickup_code:
            self.pickup_code = self.generate_pickup_code()
    
    def generate_pickup_code(self):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    
    def to_dict(self):
        return {
            'id': self.id,
            'donor_id': self.donor_id,
            'title': self.title,
            'description': self.description,
            'food_type': self.food_type,
            'quantity': self.quantity,
            'expiry_time': self.expiry_time.isoformat() if self.expiry_time else None,
            'pickup_address': self.pickup_address,
            'taluka': self.taluka,
            'area': self.area,
            'images': self.images or [],
            'status': self.status,
            'claimed_by': self.claimed_by,
            'claimed_at': self.claimed_at.isoformat() if self.claimed_at else None,
            'qr_code': self.qr_code,
            'pickup_code': self.pickup_code,
            'contact_person_name': self.contact_person_name,
            'contact_person_phone': self.contact_person_phone,
            'special_instructions': self.special_instructions,
            'gps_latitude': self.gps_latitude,
            'gps_longitude': self.gps_longitude,
            'claimed_ngo_name': self.claimed_ngo_name,
            'claimed_ngo_phone': self.claimed_ngo_phone,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'donor': self.donor.to_dict() if self.donor else None,
            'claimer': self.claimer.to_dict() if self.claimer else None
        }

class Notification(db.Model):
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    recipient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    food_id = db.Column(db.Integer, db.ForeignKey('foods.id', ondelete='CASCADE'), nullable=False)
    type = db.Column(db.Enum('food_available', 'food_claimed', 'pickup_reminder', 'food_expired', name='notification_types'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    area = db.Column(db.String(100), nullable=True)
    is_read = db.Column(db.Boolean, default=False)
    sent_via_sms = db.Column(db.Boolean, default=False)
    sent_via_app = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    recipient = db.relationship('User', foreign_keys=[recipient_id], backref='received_notifications')
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_notifications')
    food = db.relationship('Food', backref='notifications')
    
    def to_dict(self):
        return {
            'id': self.id,
            'recipient_id': self.recipient_id,
            'sender_id': self.sender_id,
            'food_id': self.food_id,
            'type': self.type,
            'title': self.title,
            'message': self.message,
            'area': self.area,
            'is_read': self.is_read,
            'sent_via_sms': self.sent_via_sms,
            'sent_via_app': self.sent_via_app,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'sender': self.sender.to_dict() if self.sender else None,
            'food': self.food.to_dict() if self.food else None
        }

class EmailLog(db.Model):
    __tablename__ = 'email_logs'
    id = db.Column(db.Integer, primary_key=True)
    ngo_email = db.Column(db.String(120), nullable=False)
    food_id = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='pending')  # sent/failed/pending
    error_message = db.Column(db.Text)
    sent_time = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'ngo_email': self.ngo_email,
            'food_id': self.food_id,
            'status': self.status,
            'error_message': self.error_message,
            'sent_time': self.sent_time.isoformat() if self.sent_time else None
        }

# Utility functions
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

def create_notification(recipient_id, sender_id, food_id, notification_type, title, message, area):
    """Create a new notification in the database"""
    try:
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

def cleanup_expired_food():
    """Background task to automatically delete expired food"""
    while True:
        try:
            with app.app_context():
                # Find ALL expired food regardless of status
                expired_foods = Food.query.filter(
                    Food.expiry_time <= datetime.utcnow()
                ).all()
                
                if expired_foods:
                    print(f"\nFound {len(expired_foods)} expired food items to clean up")
                    
                    for food in expired_foods:
                        print(f"   - Deleting: {food.title} (expired at {food.expiry_time}, status: {food.status})")
                        
                        # First delete related notifications manually
                        Notification.query.filter_by(food_id=food.id).delete()
                        
                        # Then delete the expired food
                        db.session.delete(food)
                    
                    db.session.commit()
                    print(f"Cleaned up {len(expired_foods)} expired food items\n")
                else:
                    print("✓ No expired food to clean up")
                
        except Exception as e:
            print(f"Error in cleanup_expired_food: {str(e)}")
            try:
                db.session.rollback()
            except:
                pass
        
        # Run cleanup every 5 minutes (300 seconds)
        time.sleep(300)

def start_background_tasks():
    """Start background cleanup task"""
    cleanup_thread = threading.Thread(target=cleanup_expired_food, daemon=True)
    cleanup_thread.start()
    print("Background cleanup task started - checking for expired food every 5 minutes")

# Routes
@app.route('/')
def index():
    return render_template('home_new_design.html', v=datetime.utcnow().timestamp())

@app.route('/home-new')
def home_new():
    """New homepage with video - temporary route for testing"""
    return render_template('homepage_new.html')

@app.route('/home-pro')
def home_professional():
    """Professional homepage like screenshot"""
    return render_template('home_professional.html')

@app.route('/home-final')
def home_final():
    """Final homepage with all features"""
    return render_template('home_final.html')

@app.route('/login')
def login_page():
    return render_template('auth.html')

@app.route('/register')
def register_page():
    return render_template('auth.html')

@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/auth.html')
def auth_page():
    return render_template('auth.html')

@app.route('/index_working.html')
def index_working():
    return render_template('index_working.html')

@app.route('/test')
def test():
    return '<h1>Flask App is Working!</h1><p>This is a test route.</p><p>Time: ' + str(datetime.now()) + '</p>'

@app.route('/check-food')
def check_food():
    """Check all food in database"""
    try:
        all_food = Food.query.all()
        all_users = User.query.all()
        
        html = '<h1>Food Database Check</h1><hr>'
        
        html += '<h2>All Users:</h2><ul>'
        for user in all_users:
            html += f'<li><strong>{user.name}</strong> - {user.user_type} - Area: {user.area} - Email: {user.email}</li>'
        html += '</ul><hr>'
        
        html += '<h2>All Food Items:</h2>'
        if not all_food:
            html += '<p>No food in database yet!</p>'
        else:
            for food in all_food:
                html += f'''
                <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <h3>{food.title}</h3>
                    <p><strong>Status:</strong> {food.status}</p>
                    <p><strong>Area:</strong> {food.area}</p>
                    <p><strong>Donor:</strong> {food.donor.name if food.donor else 'Unknown'}</p>
                    <p><strong>Quantity:</strong> {food.quantity}</p>
                    <p><strong>Expiry:</strong> {food.expiry_time}</p>
                    <p><strong>Pickup Code:</strong> {food.pickup_code}</p>
                    <p><strong>Created:</strong> {food.created_at}</p>
                </div>
                '''
        
        html += '<hr><p><a href="/">← Back to Homepage</a></p>'
        return html
        
    except Exception as e:
        return f'<h1>Error</h1><p>{str(e)}</p>'

@app.route('/api/admin/cleanup-expired', methods=['POST'])
def manual_cleanup_expired():
    """Manually trigger cleanup of expired food"""
    try:
        # Find ALL expired food regardless of status
        expired_foods = Food.query.filter(
            Food.expiry_time <= datetime.utcnow()
        ).all()
        
        if not expired_foods:
            return jsonify({
                'message': 'No expired food found',
                'deleted_count': 0
            }), 200
        
        deleted_items = []
        for food in expired_foods:
            deleted_items.append({
                'id': food.id,
                'title': food.title,
                'status': food.status,
                'expired_at': food.expiry_time.isoformat()
            })
            
            # First delete related notifications
            Notification.query.filter_by(food_id=food.id).delete()
            
            # Then delete the expired food
            db.session.delete(food)
        
        db.session.commit()
        
        return jsonify({
            'message': f'Successfully deleted {len(expired_foods)} expired food items',
            'deleted_count': len(expired_foods),
            'deleted_items': deleted_items
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/routes')
def list_routes():
    """List all available routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': ','.join(rule.methods),
            'path': str(rule)
        })
    
    html = '<h1>Available Routes</h1><hr>'
    html += '<h2>Pages:</h2><ul>'
    for route in sorted(routes, key=lambda x: x['path']):
        if 'GET' in route['methods'] and not route['path'].startswith('/api'):
            html += f'<li><a href="{route["path"]}" target="_blank">{route["path"]}</a> - {route["endpoint"]}</li>'
    html += '</ul>'
    
    html += '<h2>API Endpoints:</h2><ul>'
    for route in sorted(routes, key=lambda x: x['path']):
        if route['path'].startswith('/api'):
            html += f'<li><strong>{route["path"]}</strong> [{route["methods"]}] - {route["endpoint"]}</li>'
    html += '</ul>'
    
    return html

@app.route('/db-test')
def db_test():
    try:
        # Test database connection
        user_count = User.query.count()
        food_count = Food.query.count()
        notification_count = Notification.query.count()
        
        return f'''
        <h1>Database Connection Test</h1>
        <p><strong>Status:</strong> Connected Successfully</p>
        <p><strong>Database:</strong> SQLite</p>
        <p><strong>Location:</strong> instance/solapur_food_share.db</p>
        <hr>
        <h3>Table Statistics:</h3>
        <ul>
            <li>Users: {user_count}</li>
            <li>Food Items: {food_count}</li>
            <li>Notifications: {notification_count}</li>
        </ul>
        <hr>
        <p><strong>Test Time:</strong> {datetime.now()}</p>
        <p><a href="/">← Back to Homepage</a></p>
        '''
    except Exception as e:
        return f'''
        <h1>Database Connection Test</h1>
        <p><strong>Status:</strong> Error</p>
        <p><strong>Error:</strong> {str(e)}</p>
        <p><a href="/">← Back to Homepage</a></p>
        '''

@app.route('/simple')
def simple_test():
    try:
        return render_template('simple_test.html')
    except Exception as e:
        return f'<h1>Template Error</h1><p>Error: {str(e)}</p>'

@app.route('/test-expiry-countdown')
def test_expiry_countdown():
    """Test page for expiry countdown functionality"""
    try:
        with open('test_expiry_countdown.html', 'r') as f:
            return f.read()
    except Exception as e:
        return f'<h1>Test Error</h1><p>Error: {str(e)}</p>'

@app.route('/test-delete-donation')
def test_delete_donation():
    """Test page for delete donation functionality"""
    try:
        with open('test_delete_donation.html', 'r') as f:
            return f.read()
    except Exception as e:
        return f'<h1>Test Error</h1><p>Error: {str(e)}</p>'

@app.route('/debug')
def debug_info():
    import os
    template_dir = os.path.join(os.getcwd(), 'templates')
    files = os.listdir(template_dir) if os.path.exists(template_dir) else []
    
    return f'''
    <h1>Debug Info</h1>
    <p><strong>Current Directory:</strong> {os.getcwd()}</p>
    <p><strong>Template Directory:</strong> {template_dir}</p>
    <p><strong>Template Directory Exists:</strong> {os.path.exists(template_dir)}</p>
    <p><strong>Files in Templates:</strong> {files}</p>
    <p><strong>Flask Template Folder:</strong> {app.template_folder}</p>
    '''

@app.route('/test-donor')
def test_donor():
    return render_template('add_food_new.html')

@app.route('/add-food')
def add_food_page():
    from flask import make_response
    resp = make_response(render_template('add_food_new.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return resp

@app.route('/test-ngo')
def test_ngo():
    return render_template('ngo-dashboard-fixed.html')

@app.route('/test-ngo-old')
def test_ngo_old():
    return render_template('test-ngo.html', v=datetime.utcnow().timestamp())

@app.route('/ngo-dashboard-fixed')
def ngo_dashboard_fixed():
    return render_template('ngo-dashboard-fixed.html')

@app.route('/find-food')
def find_food():
    return render_template('find_food_simple.html')

@app.route('/find-food-map')
def find_food_map():
    return render_template('find_food.html')

@app.route('/route-map')
def route_map():
    return render_template('route_map.html')

@app.route('/district-food')
def district_food():
    return render_template('district_food.html')

@app.route('/api/food/detail/<int:food_id>', methods=['GET'])
def get_food_detail(food_id):
    """Get single food item details for route map"""
    try:
        food = Food.query.get(food_id)
        if not food:
            return jsonify({'message': 'Food not found'}), 404
        return jsonify({'food': food.to_dict()}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# ── NGO Contacts API (admin protected) ───────────────────────────────────────
def check_admin_password(data):
    """Verify admin password from request"""
    admin_pass = os.getenv('ADMIN_PASSWORD', 'admin@foodshare2024')
    provided = data.get('admin_password', '').strip()
    print(f"[ADMIN] Provided: '{provided}' | Expected: '{admin_pass}'")
    return provided == admin_pass

@app.route('/api/ngo-contacts', methods=['GET'])
def get_ngo_contacts():
    """Get all NGO contacts - public read"""
    try:
        from sqlalchemy import text
        rows = db.session.execute(text(
            "SELECT id, name, phone, area, taluka, active FROM ngo_contacts ORDER BY id"
        )).fetchall()
        return jsonify({'contacts': [
            {'id': r[0], 'name': r[1], 'phone': r[2], 'area': r[3], 'taluka': r[4], 'active': bool(r[5])}
            for r in rows
        ]}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/ngo-contacts', methods=['POST'])
def add_ngo_contact():
    """Add new NGO contact - admin only"""
    try:
        data = request.get_json()
        # Admin password check
        if not check_admin_password(data):
            return jsonify({'message': 'Unauthorized. Admin password required.'}), 401
        name   = data.get('name', '').strip()
        phone  = data.get('phone', '').strip()
        area   = data.get('area', '').strip()
        taluka = data.get('taluka', '').strip()
        if not name or not phone:
            return jsonify({'message': 'Name and phone required'}), 400
        from sqlalchemy import text
        db.session.execute(text(
            "INSERT INTO ngo_contacts (name, phone, area, taluka, active) VALUES (:n,:p,:a,:t,1) ON CONFLICT (phone) DO NOTHING"
        ), {'n': name, 'p': phone, 'a': area, 't': taluka or area})
        db.session.commit()
        return jsonify({'message': f'{name} added successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/ngo-contacts/<int:contact_id>', methods=['DELETE'])
def delete_ngo_contact(contact_id):
    """Delete NGO contact - admin only"""
    try:
        data = request.get_json() or {}
        if not check_admin_password(data):
            return jsonify({'message': 'Unauthorized. Admin password required.'}), 401
        from sqlalchemy import text
        db.session.execute(text("DELETE FROM ngo_contacts WHERE id=:id"), {'id': contact_id})
        db.session.commit()
        return jsonify({'message': 'Deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/admin/verify', methods=['POST'])
def verify_admin():
    """Verify admin password"""
    data = request.get_json() or {}
    if check_admin_password(data):
        return jsonify({'success': True}), 200
    return jsonify({'success': False, 'message': 'Wrong password'}), 401

@app.route('/ngo-contacts-admin')
def ngo_contacts_admin():
    return render_template('ngo_contacts_admin.html')

@app.route('/my-donations')
def my_donations():
    return render_template('my-donations.html')

@app.route('/my-claims')
def my_claims():
    return render_template('my-claims.html')

# Admin routes
@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template('admin-dashboard-ai.html')

@app.route('/admin/dashboard-old')
def admin_dashboard_old():
    return render_template('admin-dashboard.html')

@app.route('/admin/bulk-upload')
def admin_bulk_upload():
    return render_template('admin-bulk-upload.html')

@app.route('/admin/ngos')
def admin_ngos():
    return render_template('admin-ngos.html')

@app.route('/admin/ngos/add')
def admin_add_ngo():
    """Add new NGO page - redirects to bulk upload for now"""
    return render_template('admin-bulk-upload.html')

@app.route('/admin/donors')
def admin_donors():
    """View all donors"""
    return render_template('admin-ngos.html')  # Reuse NGO template for now

@app.route('/admin/food')
def admin_food():
    """View all food items"""
    return redirect('/check-food')

@app.route('/admin/reports')
def admin_reports():
    """Download reports page"""
    return render_template('admin-reports.html')

@app.route('/admin/settings')
def admin_settings():
    """Admin settings page - redirect to dashboard for now"""
    return redirect('/admin/dashboard')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    return redirect('/')

# Admin Reports API Endpoints
@app.route('/api/admin/reports/stats', methods=['GET'])
def get_admin_stats():
    """Get overall statistics"""
    try:
        total_users = User.query.count()
        total_food = Food.query.count()
        total_claimed = Food.query.filter_by(status='claimed').count() + Food.query.filter_by(status='picked_up').count()
        total_picked_up = Food.query.filter_by(status='picked_up').count()
        
        return jsonify({
            'total_users': total_users,
            'total_food': total_food,
            'total_claimed': total_claimed,
            'total_picked_up': total_picked_up
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/reports/taluka-stats', methods=['GET'])
def get_taluka_stats():
    """Get taluka-wise statistics"""
    try:
        talukas = ['Solapur City', 'Pandharpur', 'Barshi', 'Akkalkot', 'Mohol', 
                   'Malshiras', 'Karmala', 'Madha', 'Sangole', 'Mangalvedhe', 'South Solapur']
        
        stats = {}
        for taluka in talukas:
            ngos = User.query.filter_by(user_type='ngo', area=taluka).count()
            donors = User.query.filter_by(user_type='donor', area=taluka).count()
            food = Food.query.filter_by(area=taluka).count()
            
            stats[taluka] = {
                'ngos': ngos,
                'donors': donors,
                'food': food
            }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/reports/users-csv', methods=['GET'])
def download_users_csv():
    """Download all users as CSV"""
    try:
        users = User.query.all()
        
        csv_data = "ID,Name,Email,Phone,Type,Area,Created At\n"
        for user in users:
            csv_data += f"{user.id},{user.name},{user.email},{user.phone},{user.user_type},{user.area},{user.created_at}\n"
        
        from flask import Response
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=all_users.csv"}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/reports/ngos-csv', methods=['GET'])
def download_ngos_csv():
    """Download NGOs list as CSV"""
    try:
        ngos = User.query.filter_by(user_type='ngo').all()
        
        csv_data = "ID,Name,Email,Phone,Area,Organization,Registration Number,Capacity,Created At\n"
        for ngo in ngos:
            csv_data += f"{ngo.id},{ngo.name},{ngo.email},{ngo.phone},{ngo.area},{ngo.organization_name},{ngo.registration_number},{ngo.capacity},{ngo.created_at}\n"
        
        from flask import Response
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=ngos_list.csv"}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/reports/donors-csv', methods=['GET'])
def download_donors_csv():
    """Download donors list as CSV"""
    try:
        donors = User.query.filter_by(user_type='donor').all()
        
        csv_data = "ID,Name,Email,Phone,Area,Address,Created At\n"
        for donor in donors:
            csv_data += f"{donor.id},{donor.name},{donor.email},{donor.phone},{donor.area},{donor.address},{donor.created_at}\n"
        
        from flask import Response
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=donors_list.csv"}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/reports/food-csv', methods=['GET'])
def download_food_csv():
    """Download food items as CSV"""
    try:
        foods = Food.query.all()
        
        csv_data = "ID,Title,Type,Quantity,Area,Status,Donor,Expiry Time,Created At\n"
        for food in foods:
            donor_name = food.donor.name if food.donor else 'Unknown'
            csv_data += f"{food.id},{food.title},{food.food_type},{food.quantity},{food.area},{food.status},{donor_name},{food.expiry_time},{food.created_at}\n"
        
        from flask import Response
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=food_items.csv"}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-food-add')
def test_food_add():
    return render_template('test-food-add.html')

# new working donor dashboard with proper auth flow
@app.route('/working-donor')
def working_donor():
    return render_template('working-donor.html')

@app.route('/ngo-dashboard.html')
def ngo_dashboard():
    return render_template('ngo-dashboard.html')

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Check if user already exists
        existing_user = User.query.filter(
            (User.email == data['email']) | (User.phone == data['phone'])
        ).first()
        
        if existing_user:
            return jsonify({
                'message': 'User with this email or phone already exists'
            }), 400
        
        # Create new user
        user = User(
            name=data['name'],
            email=data['email'],
            phone=data['phone'],
            user_type=data['user_type'],
            taluka=data['area'],  # Use area as taluka (they are same in our case)
            area=data['area'],
            address=data['address']
        )
        
        # Add NGO specific fields if user_type is ngo
        if data['user_type'] == 'ngo':
            user.organization_name = data.get('organization_name')
            user.registration_number = data.get('registration_number')
            user.capacity = data.get('capacity')
        
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        print(f"User registered: {user.name} ({user.user_type}) in area: {user.area}")
        
        # Generate JWT token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"Registration error: {str(e)}")
        return jsonify({'message': f'Registration error: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        # Find user by email (support both plain and encrypted - use plain)
        email = data.get('email', '')
        password = data.get('password', '')

        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({'message': 'Invalid credentials'}), 400
        
        # Generate JWT token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Login error: {str(e)}'}), 500

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        return jsonify(user.to_dict()), 200
        
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/api/test-auth', methods=['GET'])
@jwt_required()
def test_auth():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        return jsonify({
            'message': 'Authentication working',
            'user_id': user_id,
            'user_name': user.name if user else 'Unknown',
            'user_type': user.user_type if user else 'Unknown',
            'user_area': user.area if user else 'Unknown'
        }), 200
    except Exception as e:
        return jsonify({'message': f'Auth test error: {str(e)}'}), 500

@app.route('/api/debug/food-add', methods=['POST'])
def debug_food_add():
    """Debug endpoint to test food addition without auth"""
    try:
        data = request.get_json()
        auth_header = request.headers.get('Authorization', 'No auth header')
        
        return jsonify({
            'message': 'Debug info',
            'received_data': data,
            'auth_header': auth_header,
            'headers': dict(request.headers)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Food routes
@app.route('/api/food/add-test', methods=['POST'])
def add_food_test():
    """Test endpoint without JWT - for debugging"""
    try:
        data = request.get_json()
        
        # Get user email from request
        user_email = data.get('user_email')
        if not user_email:
            return jsonify({'message': 'User email required'}), 400
        
        user = User.query.filter_by(email=user_email).first()

        # Auto-create guest donor if not found
        if not user:
            import random, string
            rand_phone = '9' + ''.join(random.choices(string.digits, k=9))
            donor_name = data.get('contact_person_name', user_email.split('@')[0])
            user = User(
                name=donor_name,
                email=user_email,
                phone=data.get('contact_person_phone', rand_phone),
                user_type='donor',
                address=data.get('pickup_address', '')
            )
            user.set_password('guest@123')
            db.session.add(user)
            db.session.flush()
            print(f"Auto-created guest donor: {user.name} ({user.email})")

        if user.user_type != 'donor':
            # Allow anyone to donate - change type if needed
            pass
        
        # Parse expiry time
        expiry_time = datetime.fromisoformat(data['expiry_time'].replace('Z', '+00:00'))
        
        # Create new food item
        food_data = {
            'donor_id': user.id,
            'title': data['title'],
            'description': data['description'],
            'food_type': data['food_type'],
            'quantity': data['quantity'],
            'expiry_time': expiry_time,
            'pickup_address': data['pickup_address'],
            'taluka': data.get('taluka', '') or '',
            'area': data.get('area', '') or '',
            'contact_person_name': data.get('contact_person_name', user.name),
            'contact_person_phone': data.get('contact_person_phone', user.phone),
            'special_instructions': data.get('special_instructions', '')
        }
        
        # Add GPS coordinates if provided (optional)
        if data.get('gps_latitude') and data.get('gps_longitude'):
            try:
                food_data['gps_latitude'] = float(data['gps_latitude'])
                food_data['gps_longitude'] = float(data['gps_longitude'])
            except (ValueError, TypeError):
                pass  # Ignore invalid GPS coordinates
        
        food = Food(**food_data)
        
        db.session.add(food)
        db.session.flush()
        
        # Generate QR code
        qr_data = f"FOOD_ID:{food.id}|PICKUP_CODE:{food.pickup_code}|DONOR:{user.name}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        food.qr_code = qr_code_base64
        db.session.commit()
        
        print(f"Food saved successfully with ID: {food.id}")
        
        # Notify all NGOs in the same taluka or area
        ngo_ids = set()
        if food.taluka:
            for n in User.query.filter_by(user_type='ngo', taluka=food.taluka).all():
                ngo_ids.add(n.id)
            for n in User.query.filter(User.user_type=='ngo', User.taluka==None, User.area==food.taluka).all():
                ngo_ids.add(n.id)
        if food.area:
            for n in User.query.filter_by(user_type='ngo', area=food.area).all():
                ngo_ids.add(n.id)
        ngos_in_area = User.query.filter(User.id.in_(ngo_ids)).all() if ngo_ids else []
        print(f"Found {len(ngos_in_area)} NGOs for taluka={food.taluka} area={food.area}")
        
        for ngo in ngos_in_area:
            # Create notification in database
            create_notification(
                recipient_id=ngo.id,
                sender_id=user.id,
                food_id=food.id,
                notification_type='food_available',
                title=f'New Food Available in {food.taluka or user.area}',
                message=f'{food.title} - {food.quantity} available for pickup. Expires: {food.expiry_time.strftime("%Y-%m-%d %H:%M")}',
                area=food.taluka or user.area  # Use taluka instead of village name
            )
            
            print(f"Notification sent to {ngo.name} ({ngo.email}) in {ngo.taluka or ngo.area}")

        # Send email notifications in background thread (non-blocking)
        def send_emails_bg(food_id):
            with app.app_context():
                food_obj = Food.query.get(food_id)
                if food_obj:
                    send_food_donation_emails(food_obj)

        email_thread = threading.Thread(target=send_emails_bg, args=(food.id,), daemon=True)
        email_thread.start()

        # Send SMS to NGO contacts (no login needed) in background
        def send_sms_bg(food_id):
            with app.app_context():
                food_obj = Food.query.get(food_id)
                if food_obj:
                    send_sms_to_ngo_contacts(food_obj)

        sms_thread = threading.Thread(target=send_sms_bg, args=(food.id,), daemon=True)
        sms_thread.start()

        return jsonify({
            'message': 'Food added successfully',
            'food': food.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/api/user/by-email', methods=['GET'])
def get_user_by_email():
    """Get user by email"""
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({'message': 'Email required'}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        return jsonify(user.to_dict()), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/available-in-area', methods=['GET'])
def get_food_in_area():
    """Get available food in NGO's taluka (all villages/areas under that taluka)"""
    from datetime import datetime
    
    try:
        area = request.args.get('area')
        taluka = request.args.get('taluka')
        
        if not area and not taluka:
            return jsonify({'message': 'Area or Taluka required'}), 400
        
        # If taluka is provided, get all food from that taluka
        if taluka:
            foods = Food.query.filter_by(
                taluka=taluka,
                status='available'
            ).filter(
                Food.expiry_time > datetime.utcnow()
            ).order_by(Food.created_at.desc()).all()
            
            print(f"Searching food in taluka: {taluka}")
            print(f"Found {len(foods)} available food items")
        else:
            # Fallback: search by area only
            foods = Food.query.filter_by(
                area=area,
                status='available'
            ).filter(
                Food.expiry_time > datetime.utcnow()
            ).order_by(Food.created_at.desc()).all()
            
            print(f"Searching food in area: {area}")
            print(f"Found {len(foods)} available food items")
        
        # Add AI recommendations to each food item
        foods_with_recommendations = []
        for food in foods:
            food_dict = food.to_dict()
            
            # Calculate hours remaining
            now = datetime.utcnow()
            if food.expiry_time > now:
                hours_remaining = (food.expiry_time - now).total_seconds() / 3600
                
                # Get AI recommendation
                recommendation = get_food_recommendation(
                    food.food_type, 
                    hours_remaining, 
                    food.description
                )
                
                food_dict['ai_recommendation'] = recommendation
                print(f"AI Recommendation for {food.title}: {recommendation['destination']}")
            
            foods_with_recommendations.append(food_dict)
        
        return jsonify({
            'foods': foods_with_recommendations
        }), 200
        
    except Exception as e:
        print(f"Error in get_food_in_area: {str(e)}")
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/all-available', methods=['GET'])
def get_all_available_food():
    """Get all available food across Solapur District"""
    try:
        foods = Food.query.filter_by(
            status='available'
        ).filter(
            Food.expiry_time > datetime.utcnow()
        ).order_by(Food.area, Food.created_at.desc()).all()

        areas = {}
        for food in foods:
            if food.area not in areas:
                areas[food.area] = []
            areas[food.area].append(food)

        return jsonify({
            'foods': [food.to_dict() for food in foods],
            'total': len(foods),
            'areas': list(areas.keys()),
            'by_area': {area: len(items) for area, items in areas.items()}
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/all-claimed', methods=['GET'])
def get_all_claimed_food():
    """Get all claimed food with NGO details"""
    try:
        foods = Food.query.filter_by(status='claimed').order_by(Food.claimed_at.desc()).all()
        return jsonify({'foods': [f.to_dict() for f in foods]}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/debug-all', methods=['GET'])
def debug_all_food():
    """Debug endpoint to see all food items"""
    try:
        all_foods = Food.query.all()
        print(f"\n=== ALL FOOD ITEMS IN DATABASE ===")
        for food in all_foods:
            print(f"ID: {food.id}, Title: {food.title}, Area: {food.area}, Status: {food.status}")
        
        return jsonify({
            'total': len(all_foods),
            'foods': [food.to_dict() for food in all_foods]
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/test/add-sample-food', methods=['POST'])
def add_sample_food():
    """Add sample food for testing"""
    try:
        data = request.get_json()
        area = data.get('area', 'Pandharpur')
        
        # Find a donor in that area or create test donor
        donor = User.query.filter_by(user_type='donor', area=area).first()
        
        if not donor:
            # Create test donor
            donor = User(
                name=f"Test Donor {area}",
                email=f"donor_{area.lower().replace(' ', '_')}@test.com",
                phone="9999999999",
                user_type='donor',
                area=area,
                address=f"Test Address, {area}"
            )
            donor.set_password("test123")
            db.session.add(donor)
            db.session.flush()
        
        # Create sample food
        from datetime import datetime, timedelta
        
        food = Food(
            donor_id=donor.id,
            title="Test Food - Rice & Dal",
            description="Sample food for testing. Fresh cooked rice and dal.",
            food_type="cooked",
            quantity="10 plates",
            expiry_time=datetime.utcnow() + timedelta(hours=6),
            pickup_address=f"Test Location, {area}",
            area=area,
            contact_person_name=donor.name,
            contact_person_phone=donor.phone,
            status='available'
        )
        
        db.session.add(food)
        db.session.commit()
        
        print(f"Sample food added in {area}")
        
        return jsonify({
            'message': 'Sample food added successfully',
            'food': food.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"Error adding sample food: {str(e)}")
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/claim-test', methods=['POST'])
def claim_food_test():
    """Claim food without JWT - uses ngo_contacts list"""
    try:
        data = request.get_json()
        food_id   = data.get('food_id')
        ngo_name  = data.get('ngo_name', '').strip()
        ngo_contact = data.get('ngo_contact', '').strip()

        if not food_id or not ngo_name:
            return jsonify({'message': 'Food ID and NGO name required'}), 400

        food = Food.query.get(food_id)
        if not food:
            return jsonify({'message': 'Food not found'}), 404

        if food.status != 'available':
            return jsonify({'message': 'Food is not available'}), 400

        # Claim without user FK - store NGO info directly
        food.status = 'claimed'
        food.claimed_at = datetime.utcnow()
        food.claimed_ngo_name = ngo_name
        food.claimed_ngo_phone = ngo_contact
        db.session.commit()

        return jsonify({
            'message': 'Food claimed successfully',
            'food': food.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/verify-test', methods=['POST'])
def verify_pickup_test():
    """Verify pickup without JWT"""
    try:
        data = request.get_json()
        pickup_code = data.get('pickup_code')
        ngo_email = data.get('ngo_email')
        
        if not pickup_code or not ngo_email:
            return jsonify({'message': 'Pickup code and NGO email required'}), 400
        
        ngo = User.query.filter_by(email=ngo_email).first()
        if not ngo or ngo.user_type != 'ngo':
            return jsonify({'message': 'Invalid NGO'}), 403
        
        food = Food.query.filter_by(pickup_code=pickup_code).first()
        if not food:
            return jsonify({'message': 'Invalid pickup code'}), 404
        
        if food.claimed_by != ngo.id:
            return jsonify({'message': 'You have not claimed this food'}), 403
        
        if food.status != 'claimed':
            return jsonify({'message': 'Food is not in claimed status'}), 400
        
        # Mark as picked up
        food.status = 'picked_up'
        db.session.commit()
        
        return jsonify({
            'message': 'Food pickup verified successfully',
            'food': food.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/delete/<int:food_id>', methods=['DELETE'])
def delete_food(food_id):
    """Delete a food donation (only by the donor who created it)"""
    try:
        data = request.get_json()
        user_email = data.get('user_email')
        
        if not user_email:
            return jsonify({'message': 'User email required'}), 400
        
        # Get user
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Get food item
        food = Food.query.get(food_id)
        if not food:
            return jsonify({'message': 'Food item not found'}), 404
        
        # Check if user is the donor
        if food.donor_id != user.id:
            return jsonify({'message': 'You can only delete your own donations'}), 403
        
        # Check if food is already claimed or picked up
        if food.status in ['claimed', 'picked_up']:
            return jsonify({'message': 'Cannot delete food that has been claimed or picked up'}), 400
        
        # Delete related notifications first
        Notification.query.filter_by(food_id=food_id).delete()
        
        # Delete the food item
        db.session.delete(food)
        db.session.commit()
        
        return jsonify({
            'message': 'Food donation deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get platform statistics"""
    try:
        total_donors = User.query.filter_by(user_type='donor').count()
        total_ngos = User.query.filter_by(user_type='ngo').count()
        total_food = Food.query.count()
        food_available = Food.query.filter_by(status='available').count()
        food_claimed = Food.query.filter_by(status='claimed').count()
        food_picked_up = Food.query.filter_by(status='picked_up').count()
        
        return jsonify({
            'total_donors': total_donors,
            'total_ngos': total_ngos,
            'total_food_shared': total_food,
            'food_available': food_available,
            'food_claimed': food_claimed,
            'food_delivered': food_picked_up,
            'areas_covered': len(SOLAPUR_AREAS)
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Get notifications for a user by email"""
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({'message': 'Email required'}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Get all notifications for this user, ordered by newest first
        notifications = Notification.query.filter_by(recipient_id=user.id).order_by(Notification.created_at.desc()).all()
        
        # Count unread notifications
        unread_count = Notification.query.filter_by(recipient_id=user.id, is_read=False).count()
        
        return jsonify({
            'notifications': [notif.to_dict() for notif in notifications],
            'unread_count': unread_count
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/test-ngo-fixed')
def test_ngo_fixed():
    """Fixed NGO Dashboard with correct AI recommendations"""
    return render_template('test-ngo-fixed.html')

@app.route('/test-image-upload')
def test_image_upload():
    """Simple image upload test page"""
    return render_template('test-image-upload.html')

@app.route('/debug-ngo')
def debug_ngo():
    """Debug NGO login and AI recommendations"""
    return render_template('debug_ngo.html')

@app.route('/api/notifications/mark-read', methods=['POST'])
def mark_notification_read():
    """Mark notification as read"""
    try:
        data = request.get_json()
        notification_id = data.get('notification_id')
        
        if not notification_id:
            return jsonify({'message': 'Notification ID required'}), 400
        
        notification = Notification.query.get(notification_id)
        if not notification:
            return jsonify({'message': 'Notification not found'}), 404
        
        notification.is_read = True
        db.session.commit()
        
        return jsonify({'message': 'Notification marked as read'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/notifications/mark-all-read', methods=['POST'])
def mark_all_notifications_read():
    """Mark all notifications as read for a user"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'message': 'Email required'}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Mark all notifications as read
        Notification.query.filter_by(recipient_id=user.id, is_read=False).update({'is_read': True})
        db.session.commit()
        
        return jsonify({'message': 'All notifications marked as read'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/admin/bulk-upload-ngos', methods=['POST'])
def bulk_upload_ngos():
    """Bulk upload NGOs from CSV data"""
    try:
        data = request.get_json()
        ngos = data.get('ngos', [])
        
        if not ngos:
            return jsonify({'message': 'No NGO data provided'}), 400
        
        success_count = 0
        error_count = 0
        errors = []
        
        for ngo_data in ngos:
            try:
                # Check if email already exists
                existing = User.query.filter_by(email=ngo_data.get('email')).first()
                if existing:
                    errors.append(f"Email {ngo_data.get('email')} already exists")
                    error_count += 1
                    continue
                
                # Create new NGO user
                ngo = User(
                    name=ngo_data.get('name'),
                    email=ngo_data.get('email'),
                    phone=ngo_data.get('phone'),
                    user_type='ngo',
                    area=ngo_data.get('area'),
                    address=ngo_data.get('address', ''),
                    organization_name=ngo_data.get('organization_name'),
                    registration_number=ngo_data.get('registration_number', ''),
                    capacity=int(ngo_data.get('capacity', 0)) if ngo_data.get('capacity') else None,
                    is_verified=True
                )
                
                # Set default password (NGO can change later)
                default_password = 'ngo123'  # You can generate random passwords
                ngo.set_password(default_password)
                
                db.session.add(ngo)
                success_count += 1
                
            except Exception as e:
                errors.append(f"Error with {ngo_data.get('email')}: {str(e)}")
                error_count += 1
                continue
        
        db.session.commit()
        
        return jsonify({
            'message': 'Bulk upload completed',
            'success_count': success_count,
            'error_count': error_count,
            'errors': errors
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Bulk upload error: {str(e)}'}), 500

@app.route('/api/admin/ngos', methods=['GET'])
def get_all_ngos():
    """Get all NGOs for admin panel"""
    try:
        ngos = User.query.filter_by(user_type='ngo').order_by(User.created_at.desc()).all()
        return jsonify({
            'ngos': [ngo.to_dict() for ngo in ngos]
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/admin/ngos/<int:ngo_id>', methods=['DELETE'])
def delete_ngo(ngo_id):
    """Delete an NGO"""
    try:
        ngo = User.query.get(ngo_id)
        if not ngo or ngo.user_type != 'ngo':
            return jsonify({'message': 'NGO not found'}), 404
        
        db.session.delete(ngo)
        db.session.commit()
        
        return jsonify({'message': 'NGO deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/donor/my-donations', methods=['GET'])
def get_my_donations():
    """Get donor's donations by email or phone"""
    try:
        email = request.args.get('email', '').strip()
        phone = request.args.get('phone', '').strip()

        if not email and not phone:
            return jsonify({'donations': []}), 200

        user = None

        # Try phone first
        if phone:
            phone_last10 = phone[-10:]
            # Try direct contact_person_phone match in foods
            foods = Food.query.filter(
                Food.contact_person_phone.like('%' + phone_last10)
            ).order_by(Food.created_at.desc()).all()
            if foods:
                return jsonify({'donations': [f.to_dict() for f in foods]}), 200
            # Try user table
            user = User.query.filter(User.phone.like('%' + phone_last10)).first()

        # Try email
        if not user and email:
            user = User.query.filter_by(email=email).first()

        if not user:
            return jsonify({'donations': []}), 200

        foods = Food.query.filter_by(donor_id=user.id).order_by(Food.created_at.desc()).all()
        return jsonify({'donations': [food.to_dict() for food in foods]}), 200

    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/ngo/my-claims', methods=['GET'])
def get_my_claims():
    """Get claims by ngo_name or ngo_phone"""
    try:
        ngo_name  = request.args.get('ngo_name', '').strip()
        ngo_phone = request.args.get('ngo_phone', '').strip()

        if not ngo_name and not ngo_phone:
            return jsonify({'foods': [], 'claims': []}), 200

        from sqlalchemy import or_
        query = Food.query.filter(Food.status == 'claimed')
        if ngo_name and ngo_phone:
            query = query.filter(or_(
                Food.claimed_ngo_name == ngo_name,
                Food.claimed_ngo_phone == ngo_phone
            ))
        elif ngo_name:
            query = query.filter(Food.claimed_ngo_name == ngo_name)
        else:
            query = query.filter(Food.claimed_ngo_phone == ngo_phone)

        foods = query.order_by(Food.claimed_at.desc()).all()
        return jsonify({
            'claims': [f.to_dict() for f in foods],
            'foods':  [f.to_dict() for f in foods]
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/food/add', methods=['POST'])
@jwt_required()
def add_food():
    try:
        user_id = get_jwt_identity()
        
        # Convert to int if it's a string
        if isinstance(user_id, str):
            try:
                user_id = int(user_id)
            except:
                pass
        
        user = User.query.get(user_id)
        
        print(f"=== FOOD ADD REQUEST ===")
        print(f"User ID from JWT: {user_id} (type: {type(user_id)})")
        print(f"User found: {user.name if user else 'NOT FOUND'}")
        print(f"User Type: {user.user_type if user else 'N/A'}")
        
        if not user:
            print("ERROR: User not found in database")
            return jsonify({'message': 'User not found'}), 404
        
        if user.user_type != 'donor':
            print(f"ERROR: User is {user.user_type}, not donor")
            return jsonify({'message': 'Only donors can add food'}), 403
        
        data = request.get_json()
        print(f"Received data keys: {list(data.keys())}")
        print(f"Images in data: {'images' in data}")
        if 'images' in data:
            print(f"Images data type: {type(data['images'])}")
            print(f"Images length: {len(data['images'])}")
            print(f"First image type: {type(data['images'][0]) if data['images'] else 'None'}")
        print(f"Received data: {data}")
        
        # Validate required fields
        required_fields = ['title', 'description', 'food_type', 'quantity', 'container_type', 'expiry_time', 'pickup_address']
        for field in required_fields:
            if field not in data or not data[field]:
                print(f"ERROR: Missing required field: {field}")
                return jsonify({'message': f'Missing required field: {field}'}), 400
        
        # Parse expiry time
        try:
            expiry_time = datetime.fromisoformat(data['expiry_time'].replace('Z', '+00:00'))
            print(f"Parsed expiry time: {expiry_time}")
        except Exception as e:
            print(f"ERROR: Invalid expiry time format: {str(e)}")
            return jsonify({'message': f'Invalid expiry time format: {str(e)}'}), 400

        # Dynamic expiry prediction (override if prediction data provided)
        storage_condition = data.get('storage_condition', 'room_temperature')
        preparation_time_str = data.get('preparation_time')
        current_temperature = data.get('current_temperature')
        if current_temperature:
            try:
                current_temperature = float(current_temperature)
            except:
                current_temperature = None

        if storage_condition and preparation_time_str:
            try:
                predicted_expiry, predicted_hours = predict_food_expiry(
                    food_type=data['food_type'],
                    container_type=data['container_type'],
                    storage_condition=storage_condition,
                    preparation_time=preparation_time_str,
                    current_temperature=current_temperature
                )
                expiry_time = predicted_expiry
                print(f"[EXPIRY] Predicted expiry overridden: {expiry_time} ({predicted_hours}h)")
            except Exception as e:
                print(f"[EXPIRY] Prediction failed, using donor-entered time: {e}")

        # Create new food item
        food = Food(
            donor_id=user_id,
            title=data['title'],
            description=data['description'],
            food_type=data['food_type'],
            quantity=data['quantity'],
            container_type=data['container_type'],
            expiry_time=expiry_time,
            pickup_address=data['pickup_address'],
            area=user.area,
            contact_person_name=data.get('contact_person_name', user.name),
            contact_person_phone=data.get('contact_person_phone', user.phone),
            special_instructions=data.get('special_instructions', '')
        )
        
        print(f"Created food object: {food.title}")
        
        db.session.add(food)
        db.session.flush()
        
        print(f"Food ID: {food.id}, Pickup Code: {food.pickup_code}")
        
        # Generate QR code
        qr_data = f"FOOD_ID:{food.id}|PICKUP_CODE:{food.pickup_code}|DONOR:{user.name}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        food.qr_code = qr_code_base64
        db.session.commit()
        
        print(f"Food saved successfully with ID: {food.id}")
        
        # Send email notifications to NGOs in the same taluka
        send_food_donation_emails(food)
        
        # Notify all NGOs in the same area
        ngos_in_area = User.query.filter_by(user_type='ngo', area=user.area).all()
        print(f"Found {len(ngos_in_area)} NGOs in area {user.area}")
        
        for ngo in ngos_in_area:
            # Create notification
            create_notification(
                recipient_id=ngo.id,
                sender_id=user_id,
                food_id=food.id,
                notification_type='food_available',
                title=f'New Food Available in {user.area}',
                message=f'{food.title} - {food.quantity} available for pickup. Expires: {food.expiry_time.strftime("%Y-%m-%d %H:%M")}',
                area=user.area
            )
            
            # Send SMS notification
            send_sms_notification(
                phone=ngo.phone,
                message=f'Food Alert: {food.title} available in {user.area}. Pickup code: {food.pickup_code}. Contact: {food.contact_person_phone}'
            )
        
        print("=== FOOD ADD SUCCESS ===")
        
        return jsonify({
            'message': 'Food added successfully',
            'food': food.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"=== FOOD ADD ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'message': f'Error adding food: {str(e)}'}), 500

@app.route('/api/food/available', methods=['GET'])
@jwt_required()
def get_available_food():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        # Get available food in user's area
        foods = Food.query.filter_by(
            area=user.area,
            status='available'
        ).filter(Food.expiry_time > datetime.utcnow()).all()
        
        return jsonify({
            'foods': [food.to_dict() for food in foods]
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching food: {str(e)}'}), 500

@app.route('/api/food/claim/<int:food_id>', methods=['POST'])
@jwt_required()
def claim_food(food_id):
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'ngo':
            return jsonify({'message': 'Only NGOs can claim food'}), 403
        
        food = Food.query.get(food_id)
        if not food:
            return jsonify({'message': 'Food not found'}), 404
        
        if food.status != 'available':
            return jsonify({'message': 'Food is not available'}), 400
        
        if food.area != user.area:
            return jsonify({'message': 'Food is not in your area'}), 403
        
        # Claim the food
        food.status = 'claimed'
        food.claimed_by = user_id
        food.claimed_at = datetime.utcnow()
        
        db.session.commit()
        
        # Notify donor
        create_notification(
            recipient_id=food.donor_id,
            sender_id=user_id,
            food_id=food_id,
            notification_type='food_claimed',
            title='Your Food Has Been Claimed',
            message=f'{user.organization_name or user.name} has claimed your food: {food.title}',
            area=food.area
        )
        
        # Send SMS to donor
        donor = User.query.get(food.donor_id)
        send_sms_notification(
            phone=donor.phone,
            message=f'Your food "{food.title}" has been claimed by {user.organization_name or user.name}. Pickup code: {food.pickup_code}'
        )
        
        return jsonify({
            'message': 'Food claimed successfully',
            'food': food.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error claiming food: {str(e)}'}), 500

@app.route('/api/food/verify-pickup', methods=['POST'])
@jwt_required()
def verify_pickup():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'ngo':
            return jsonify({'message': 'Only NGOs can verify pickup'}), 403
        
        data = request.get_json()
        pickup_code = data.get('pickup_code')
        
        food = Food.query.filter_by(pickup_code=pickup_code).first()
        if not food:
            return jsonify({'message': 'Invalid pickup code'}), 404
        
        if food.claimed_by != user_id:
            return jsonify({'message': 'You have not claimed this food'}), 403
        
        if food.status != 'claimed':
            return jsonify({'message': 'Food is not in claimed status'}), 400
        
        # Mark as picked up
        food.status = 'picked_up'
        db.session.commit()
        
        # Notify donor
        create_notification(
            recipient_id=food.donor_id,
            sender_id=user_id,
            food_id=food.id,
            notification_type='pickup_reminder',
            title='Food Successfully Picked Up',
            message=f'{user.organization_name or user.name} has picked up your food: {food.title}',
            area=food.area
        )
        
        return jsonify({
            'message': 'Food pickup verified successfully',
            'food': food.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error verifying pickup: {str(e)}'}), 500

# Email Log Management
@app.route('/api/admin/email-logs', methods=['GET'])
def get_email_logs():
    """Get all email logs for admin"""
    try:
        logs = EmailLog.query.order_by(EmailLog.sent_time.desc()).all()
        return jsonify({
            'logs': [log.to_dict() for log in logs]
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/admin/email-logs/<int:log_id>', methods=['GET'])
def get_email_log(log_id):
    """Get specific email log"""
    try:
        log = EmailLog.query.get(log_id)
        if not log:
            return jsonify({'message': 'Email log not found'}), 404
        
        return jsonify({
            'log': log.to_dict()
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/admin/email-stats', methods=['GET'])
def get_email_stats():
    """Get email statistics"""
    try:
        total_sent = EmailLog.query.filter_by(status='sent').count()
        total_failed = EmailLog.query.filter_by(status='failed').count()
        total_pending = EmailLog.query.filter_by(status='pending').count()
        
        # Get today's stats
        today = datetime.utcnow().date()
        today_sent = EmailLog.query.filter_by(status='sent').filter(
            EmailLog.sent_time >= today
        ).count()
        today_failed = EmailLog.query.filter_by(status='failed').filter(
            EmailLog.sent_time >= today
        ).count()
        
        return jsonify({
            'total_sent': total_sent,
            'total_failed': total_failed,
            'total_pending': total_pending,
            'today_sent': today_sent,
            'today_failed': today_failed,
            'success_rate': (total_sent / (total_sent + total_failed) * 100) if (total_sent + total_failed) > 0 else 0
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# AI Food Quality Check
@app.route('/api/verify-food-quality', methods=['POST'])
def verify_food_quality():
    """Verify food quality using REAL AI image analysis (OpenCV)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        image_bytes = image_file.read()
        print(f"Image received: {len(image_bytes)} bytes, filename: {image_file.filename}")
        
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty image file received'}), 400
        
        # Inline AI analysis - PIL + OpenCV directly (no external module cache issues)
        ai_result = _analyze_food_image_inline(image_bytes)
        
        print(f"=== REAL AI ANALYSIS RESULT ===")
        print(f"Quality: {ai_result.get('quality')}")
        print(f"Confidence: {ai_result.get('confidence')}")
        print(f"Message: {ai_result.get('message')}")
        if 'metrics' in ai_result:
            m = ai_result['metrics']
            print(f"Dark rot ratio: {m.get('dark_rot_ratio', 0):.3f}")
            print(f"Dark brown ratio: {m.get('dark_brown_ratio', 0):.3f}")
            print(f"Mold ratio: {m.get('mold_ratio', 0):.3f}")
            print(f"Patch spoilage: {m.get('patch_spoilage', 0):.3f}")
        
        # Map quality to frontend expected values
        raw_quality = ai_result.get('quality', 'poor')
        if raw_quality == 'fresh':
            quality = 'good'
            suggestions = [
                "Food looks suitable for immediate distribution",
                "Maintain proper temperature until pickup",
                "Consider packaging for safe transport"
            ]
        elif raw_quality == 'unclear':
            quality = 'poor'
            suggestions = [
                "This does not appear to be a food image",
                "Please upload a clear photo of the food you want to donate",
                "Supported: cooked food, fruits, vegetables, packaged food"
            ]
        elif raw_quality == 'moderate':
            quality = 'fair'
            suggestions = [
                "Check for any signs of spoilage before distribution",
                "Consider faster distribution timeline",
                "Inspect food quality upon pickup"
            ]
        else:  # poor / error
            quality = 'poor'
            if ai_result.get('quality') == 'error':
                suggestions = [
                    "Could not analyze image clearly - please try with better lighting",
                    "Ensure food is clearly visible in the photo",
                    "You can still proceed with donation"
                ]
            else:
                suggestions = [
                    "Food shows signs of spoilage - black rot, mold, or discoloration detected",
                    "NOT recommended for human consumption",
                    "Consider composting or proper disposal"
                ]
        
        # Normalize confidence to 0-1 range
        raw_confidence = ai_result.get('confidence', 0)
        confidence = raw_confidence / 100.0 if raw_confidence > 1 else raw_confidence
        
        return jsonify({
            'quality': quality,
            'confidence': round(confidence, 2),
            'message': ai_result.get('message', 'Analysis complete'),
            'suggestions': suggestions
        })
        
    except Exception as e:
        print(f"AI Analysis Error: {str(e)}")
        return jsonify({
            'quality': 'error',
            'confidence': 0,
            'message': f'AI analysis failed: {str(e)}'
        }), 500

@app.route('/api/food/<int:food_id>/expired', methods=['POST'])
def handle_food_expired(food_id):
    """Handle when food expires - delete from DB"""
    try:
        food = Food.query.get(food_id)
        if not food:
            return jsonify({'message': 'Food not found'}), 404

        # Only delete if available (not claimed/picked up)
        if food.status == 'available':
            # Delete related notifications first
            Notification.query.filter_by(food_id=food_id).delete()
            db.session.delete(food)
            db.session.commit()
            print(f"[EXPIRED] Food {food_id} '{food.title}' deleted automatically")
            return jsonify({'message': 'Food deleted', 'deleted': True}), 200
        else:
            return jsonify({'message': f'Food status is {food.status}, not deleted', 'deleted': False}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error handling expiry: {str(e)}'}), 500

# ============================================================
# DYNAMIC FOOD EXPIRY PREDICTION SYSTEM
# ============================================================

def predict_food_expiry(food_type, container_type, storage_condition, preparation_time, current_temperature=None):
    """Predict food expiry time based on multiple factors."""
    base_hours = {
        'cooked': 4, 'rice': 4, 'curry': 3, 'bread': 6,
        'bakery': 6, 'dairy': 4, 'fruits': 8, 'vegetables': 12,
        'raw': 12, 'packaged': 48, 'other': 4
    }
    hours = base_hours.get(food_type, 4)
    
    # Storage condition: refrigerated = 3x longer
    if storage_condition == 'refrigerated':
        hours = hours * 3
    
    # Container type effect
    container_multipliers = {
        'plastic': 0.8, 'steel': 1.0, 'glass': 1.0,
        'aluminum': 0.9, 'paper': 0.7, 'thermocol': 0.7, 'other': 0.9
    }
    hours = hours * container_multipliers.get(container_type, 0.9)
    
    # Temperature effect
    if current_temperature:
        if current_temperature > 30:
            hours = hours * 0.75
        elif current_temperature > 25:
            hours = hours * 0.9
    
    if isinstance(preparation_time, str):
        try:
            preparation_time = datetime.fromisoformat(preparation_time.replace('Z', '+00:00'))
        except:
            preparation_time = datetime.utcnow()
    
    predicted_expiry = preparation_time + timedelta(hours=hours)
    print(f"[EXPIRY PREDICTION] type={food_type}, container={container_type}, storage={storage_condition}, hours={hours:.2f}")
    return predicted_expiry, round(hours, 2)


@app.route('/api/food/predict-expiry', methods=['POST'])
def predict_expiry_api():
    """API endpoint to predict food expiry time"""
    try:
        data = request.get_json()
        food_type = data.get('food_type', 'cooked')
        container_type = data.get('container_type', 'other')
        storage_condition = data.get('storage_condition', 'room_temperature')
        preparation_time = data.get('preparation_time', datetime.utcnow().isoformat())
        current_temperature = data.get('current_temperature')
        
        predicted_expiry, hours = predict_food_expiry(
            food_type, container_type, storage_condition, preparation_time, current_temperature
        )
        return jsonify({
            'predicted_expiry': predicted_expiry.isoformat(),
            'safe_hours': hours,
            'message': f'Food will be safe for approximately {hours:.1f} hours'
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500


# AI Food Recommendation System
def get_food_recommendation(food_type, hours_remaining, description=""):
    """AI-based recommendation for food distribution"""
    try:
        print(f"=== AI RECOMMENDATION START ===")
        print(f"Food Type: {food_type}")
        print(f"Hours Remaining: {hours_remaining}")
        
        # Priority rules based on food type and freshness
        recommendation = {
            "destination": "",
            "priority": "",
            "reason": "",
            "confidence": 0.0,
            "action_needed": "",
            "timeline": {
                "human_safe_hours": 0,
                "animal_safe_hours": 0,
                "current_hours_remaining": hours_remaining,
                "human_safe_until": "",
                "animal_safe_until": "",
                "time_status": ""
            }
        }
        
        # Calculate timeline based on food type
        if food_type == 'cooked':
            human_safe_hours = 4
            animal_safe_hours = 8
        elif food_type == 'raw':
            human_safe_hours = 8
            animal_safe_hours = 12
        elif food_type == 'packaged':
            human_safe_hours = 24
            animal_safe_hours = 48
        else:  # other
            human_safe_hours = 6
            animal_safe_hours = 10
        
        # Calculate safe until times
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        
        # Human safe until (minimum of human safe hours and remaining time)
        human_safe_until = now + timedelta(hours=min(human_safe_hours, hours_remaining))
        recommendation["timeline"]["human_safe_hours"] = min(human_safe_hours, hours_remaining)
        recommendation["timeline"]["human_safe_until"] = human_safe_until.strftime("%Y-%m-%d %H:%M")
        
        # Animal safe until (minimum of animal safe hours and remaining time)
        animal_safe_until = now + timedelta(hours=min(animal_safe_hours, hours_remaining))
        recommendation["timeline"]["animal_safe_hours"] = min(animal_safe_hours, hours_remaining)
        recommendation["timeline"]["animal_safe_until"] = animal_safe_until.strftime("%Y-%m-%d %H:%M")
        
        # Determine time status
        if hours_remaining > human_safe_hours:
            recommendation["timeline"]["time_status"] = "Safe for Humans"
        elif hours_remaining > animal_safe_hours:
            recommendation["timeline"]["time_status"] = "Safe for Animals Only"
        else:
            recommendation["timeline"]["time_status"] = "Expiring Soon"
        
        # Rule 1: Cooked food - highest priority for humans if fresh
        if food_type == 'cooked':
            if hours_remaining > 4:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "High Priority"
                recommendation["reason"] = "Fresh cooked food is perfect for human consumption"
                recommendation["confidence"] = 0.95
                recommendation["action_needed"] = "Immediate distribution recommended"
            elif hours_remaining > 2:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "Medium Priority"
                recommendation["reason"] = "Cooked food still suitable for humans, but serve quickly"
                recommendation["confidence"] = 0.85
                recommendation["action_needed"] = "Urgent distribution needed"
            else:
                recommendation["destination"] = "Animals"
                recommendation["priority"] = "Low Priority"
                recommendation["reason"] = "Cooked food no longer safe for humans due to time constraints"
                recommendation["confidence"] = 0.90
                recommendation["action_needed"] = "Quick distribution to animals required"
        
        # Rule 2: Raw food (fruits, vegetables)
        elif food_type == 'raw':
            if hours_remaining > 8:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "High Priority"
                recommendation["reason"] = "Fresh raw food ideal for human consumption"
                recommendation["confidence"] = 0.90
                recommendation["action_needed"] = "Perfect for immediate human distribution"
            elif hours_remaining > 4:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "Medium Priority"
                recommendation["reason"] = "Raw food still good for humans with quick processing"
                recommendation["confidence"] = 0.80
                recommendation["action_needed"] = "Process and distribute quickly to humans"
            elif hours_remaining > 2:
                recommendation["destination"] = "Animals"
                recommendation["priority"] = "Medium Priority"
                recommendation["reason"] = "Raw food past ideal time for humans but good for animals"
                recommendation["confidence"] = 0.85
                recommendation["action_needed"] = "Suitable for animal feed"
            else:
                recommendation["destination"] = "Compost/Waste"
                recommendation["priority"] = "Low Priority"
                recommendation["reason"] = "Raw food no longer suitable for consumption"
                recommendation["confidence"] = 0.95
                recommendation["action_needed"] = "Dispose as compost waste"
        
        # Rule 3: Packaged food
        elif food_type == 'packaged':
            if hours_remaining > 12:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "High Priority"
                recommendation["reason"] = "Packaged food with long shelf life, perfect for humans"
                recommendation["confidence"] = 0.95
                recommendation["action_needed"] = "Store and distribute to humans as needed"
            elif hours_remaining > 6:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "Medium Priority"
                recommendation["reason"] = "Packaged food still good for human consumption"
                recommendation["confidence"] = 0.85
                recommendation["action_needed"] = "Distribute to humans within normal timeframe"
            else:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "Low Priority"
                recommendation["reason"] = "Packaged food approaching expiry but still safe for humans"
                recommendation["confidence"] = 0.75
                recommendation["action_needed"] = "Quick distribution to humans recommended"
        
        # Rule 4: Other food types
        else:
            if hours_remaining > 6:
                recommendation["destination"] = "Humans"
                recommendation["priority"] = "Medium Priority"
                recommendation["reason"] = "General food suitable for human consumption"
                recommendation["confidence"] = 0.70
                recommendation["action_needed"] = "Standard distribution to humans"
            elif hours_remaining > 3:
                recommendation["destination"] = "Animals"
                recommendation["priority"] = "Low Priority"
                recommendation["reason"] = "Food better suited for animal consumption"
                recommendation["confidence"] = 0.80
                recommendation["action_needed"] = "Consider for animal feed"
            else:
                recommendation["destination"] = "Compost/Waste"
                recommendation["priority"] = "Very Low Priority"
                recommendation["reason"] = "Food no longer suitable for any consumption"
                recommendation["confidence"] = 0.90
                recommendation["action_needed"] = "Dispose as waste"
        
        # Check for keywords in description that might affect recommendation
        description_lower = description.lower()
        if any(word in description_lower for word in ['spoiled', 'bad', 'rotten', 'expired']):
            recommendation["destination"] = "Compost/Waste"
            recommendation["priority"] = "Very Low Priority"
            recommendation["reason"] = "Food appears to be spoiled based on description"
            recommendation["confidence"] = 0.95
            recommendation["action_needed"] = "Immediate disposal required"
        elif any(word in description_lower for word in ['fresh', 'new', 'just made']):
            recommendation["confidence"] = min(1.0, recommendation["confidence"] + 0.1)
            recommendation["reason"] += " (Confirmed fresh from description)"
        
        print(f"=== AI RECOMMENDATION COMPLETE ===")
        print(f"Destination: {recommendation['destination']}")
        print(f"Priority: {recommendation['priority']}")
        print(f"Confidence: {recommendation['confidence']}")
        
        return recommendation
        
    except Exception as e:
        print(f"AI Recommendation Error: {str(e)}")
        return {
            "destination": "Humans",
            "priority": "Medium Priority",
            "reason": "AI analysis failed, defaulting to human consumption",
            "confidence": 0.5,
            "action_needed": "Manual inspection recommended"
        }

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('join_area')
def handle_join_area(data):
    area = data['area']
    join_room(area)
    print(f'Client {request.sid} joined area: {area}')

@socketio.on('leave_area')
def handle_leave_area(data):
    area = data['area']
    leave_room(area)
    print(f'Client {request.sid} left area: {area}')

# Create database tables
with app.app_context():
    db.create_all()

# ── Translation API ──────────────────────────────────────────────────────────
@app.route('/api/translate', methods=['POST'])
def translate_texts():
    """Translate texts using Google Translate unofficial API"""
    import requests as req_lib
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        lang  = data.get('lang', 'hi')

        if not texts:
            return jsonify({'translated': []}), 200

        translated = []
        for text in texts:
            t = text.strip()
            if not t or len(t) < 2 or t.isdigit():
                translated.append(text)
                continue
            try:
                url = 'https://translate.googleapis.com/translate_a/single'
                params = {
                    'client': 'gtx',
                    'sl': 'en',
                    'tl': lang,
                    'dt': 't',
                    'q': t[:500]
                }
                r = req_lib.get(url, params=params, timeout=5)
                result = r.json()
                out = ''.join([s[0] for s in result[0] if s[0]])
                translated.append(out if out else text)
            except Exception as ex:
                translated.append(text)

        return jsonify({'translated': translated}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'translated': data.get('texts', [])}), 200

# Start background cleanup task
start_background_tasks()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
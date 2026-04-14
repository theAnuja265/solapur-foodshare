from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from datetime import datetime
import qrcode
import io
import base64

food_bp = Blueprint('food', __name__)

@food_bp.route('/add', methods=['POST'])
@jwt_required()
def add_food():
    try:
        from models import Food, User, Notification
        from utils.notifications import send_sms_notification, create_notification
        
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'donor':
            return jsonify({'message': 'Only donors can add food'}), 403
        
        data = request.get_json()
        
        # Create new food item
        food = Food(
            donor_id=user_id,
            title=data['title'],
            description=data['description'],
            food_type=data['food_type'],
            quantity=data['quantity'],
            expiry_time=datetime.fromisoformat(data['expiry_time'].replace('Z', '+00:00')),
            pickup_address=data['pickup_address'],
            area=user.area,  # Use donor's area
            contact_person_name=data.get('contact_person_name', user.name),
            contact_person_phone=data.get('contact_person_phone', user.phone),
            special_instructions=data.get('special_instructions', '')
        )
        
        db.session.add(food)
        db.session.flush()  # Get the food ID
        
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
        
        # Notify all NGOs in the same area
        ngos_in_area = User.query.filter_by(user_type='ngo', area=user.area).all()
        
        for ngo in ngos_in_area:
            # Create notification
            notification = create_notification(
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
        
        return jsonify({
            'message': 'Food added successfully',
            'food': food.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error adding food: {str(e)}'}), 500

@food_bp.route('/available', methods=['GET'])
@jwt_required()
def get_available_food():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        print(f"DEBUG: User area: {user.area}")
        print(f"DEBUG: User taluka: {getattr(user, 'taluka', 'No taluka')}")
        
        # Get all available food for debugging
        all_foods = Food.query.filter_by(status='available').filter(Food.expiry_time > datetime.utcnow()).all()
        
        print(f"DEBUG: Found {len(all_foods)} total available foods")
        
        return jsonify({
            'foods': [food.to_dict() for food in all_foods],
            'debug_info': {
                'user_area': user.area,
                'user_taluka': getattr(user, 'taluka', 'No taluka'),
                'total_available_count': len(all_foods)
            }
        }), 200
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({'message': f'Error fetching food: {str(e)}'}), 500

@food_bp.route('/claim/<int:food_id>', methods=['POST'])
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
        notification = create_notification(
            recipient_id=food.donor_id,
            sender_id=user_id,
            food_id=food_id,
            notification_type='food_claimed',
            title='Your Food Has Been Claimed',
            message=f'{user.organization_name} has claimed your food: {food.title}',
            area=food.area
        )
        
        # Send SMS to donor
        donor = User.query.get(food.donor_id)
        send_sms_notification(
            phone=donor.phone,
            message=f'Your food "{food.title}" has been claimed by {user.organization_name}. Pickup code: {food.pickup_code}'
        )
        
        return jsonify({
            'message': 'Food claimed successfully',
            'food': food.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error claiming food: {str(e)}'}), 500

@food_bp.route('/verify-pickup', methods=['POST'])
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
        notification = create_notification(
            recipient_id=food.donor_id,
            sender_id=user_id,
            food_id=food.id,
            notification_type='pickup_reminder',
            title='Food Successfully Picked Up',
            message=f'{user.organization_name} has picked up your food: {food.title}',
            area=food.area
        )
        
        return jsonify({
            'message': 'Food pickup verified successfully',
            'food': food.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error verifying pickup: {str(e)}'}), 500

@food_bp.route('/my-donations', methods=['GET'])
@jwt_required()
def get_my_donations():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'donor':
            return jsonify({'message': 'Only donors can view donations'}), 403
        
        foods = Food.query.filter_by(donor_id=user_id).order_by(Food.created_at.desc()).all()
        
        return jsonify({
            'foods': [food.to_dict() for food in foods]
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching donations: {str(e)}'}), 500

@food_bp.route('/my-claims', methods=['GET'])
@jwt_required()
def get_my_claims():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'ngo':
            return jsonify({'message': 'Only NGOs can view claims'}), 403
        
        foods = Food.query.filter_by(claimed_by=user_id).order_by(Food.claimed_at.desc()).all()
        
        return jsonify({

@food_bp.route('/claim/<int:food_id>', methods=['POST'])
@jwt_required()
def claim_food(food_id):
    try:
        from models import Food, User
        from utils.notifications import send_sms_notification, create_notification
        
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'ngo':
            return jsonify({'message': 'Only NGOs can claim food'}), 403
        
        food = Food.query.get(food_id)
        if not food:
            return jsonify({'message': 'Food not found'}), 404
        
        if food.status != 'available':
            return jsonify({'message': 'Food is not available'}), 400
        
        if food.area != request.args.get('area', user.area):
            return jsonify({'message': 'Food is not in your area'}), 403
        
        # Claim the food
        food.status = 'claimed'
        food.claimed_by = user_id
        food.claimed_at = datetime.utcnow()
        
        db.session.commit()
        
        # Notify donor
        notification = create_notification(
            recipient_id=food.donor_id,
            sender_id=user_id,
            food_id=food_id,
            notification_type='food_claimed',
            title='Your Food Has Been Claimed',
            message=f'{user.organization_name} has claimed your food: {food.title}',
            area=food.area
        )
        
        # Send SMS to donor
        donor = User.query.get(food.donor_id)
        send_sms_notification(
            phone=donor.phone,
            message=f'Your food "{food.title}" has been claimed by {user.organization_name}. Pickup code: {food.pickup_code}'
        )
        
        return jsonify({
            'message': 'Food claimed successfully',
            'food': food.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error claiming food: {str(e)}'}), 500

@food_bp.route('/verify-pickup', methods=['POST'])
@jwt_required()
def verify_pickup():
    try:
        from models import Food, User
        from utils.notifications import create_notification
        
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
        notification = create_notification(
            recipient_id=food.donor_id,
            sender_id=user_id,
            food_id=food.id,
            notification_type='pickup_reminder',
            title='Food Successfully Picked Up',
            message=f'{user.organization_name} has picked up your food: {food.title}',
            area=food.area
        )
        
        return jsonify({
            'message': 'Food pickup verified successfully',
            'food': food.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error verifying pickup: {str(e)}'}), 500

@food_bp.route('/my-donations', methods=['GET'])
@jwt_required()
def get_my_donations():
    try:
        from models import Food, User
        
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'donor':
            return jsonify({'message': 'Only donors can view donations'}), 403
        
        foods = Food.query.filter_by(donor_id=user_id).order_by(Food.created_at.desc()).all()
        
        return jsonify({
            'foods': [food.to_dict() for food in foods]
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching donations: {str(e)}'}), 500

@food_bp.route('/my-claims', methods=['GET'])
@jwt_required()
def get_my_claims():
    try:
        from models import Food, User
        
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if user.user_type != 'ngo':
            return jsonify({'message': 'Only NGOs can view claims'}), 403
        
        foods = Food.query.filter_by(claimed_by=user_id).order_by(Food.claimed_at.desc()).all()
        
        return jsonify({
            'foods': [food.to_dict() for food in foods]
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching claims: {str(e)}'}), 500
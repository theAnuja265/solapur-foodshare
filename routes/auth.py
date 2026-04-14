from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    try:
        # Import here to avoid circular imports
        from models import User
        
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
        
        # Generate JWT token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Registration error: {str(e)}'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        from models import User
        
        data = request.get_json()
        
        # Find user by email
        user = User.query.filter_by(email=data['email']).first()
        
        if not user or not user.check_password(data['password']):
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

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        from models import User
        
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        return jsonify(user.to_dict()), 200
        
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db

notifications_bp = Blueprint('notifications', __name__)

@notifications_bp.route('/', methods=['GET'])
@jwt_required()
def get_notifications():
    try:
        from models import Notification
        
        user_id = get_jwt_identity()
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        
        # Build query
        query = Notification.query.filter_by(recipient_id=user_id)
        
        if unread_only:
            query = query.filter_by(is_read=False)
        
        # Paginate results
        notifications = query.order_by(Notification.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'notifications': [notification.to_dict() for notification in notifications.items],
            'total': notifications.total,
            'pages': notifications.pages,
            'current_page': page,
            'per_page': per_page,
            'has_next': notifications.has_next,
            'has_prev': notifications.has_prev
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching notifications: {str(e)}'}), 500

@notifications_bp.route('/mark-read/<int:notification_id>', methods=['PUT'])
@jwt_required()
def mark_notification_read(notification_id):
    try:
        from models import Notification
        
        user_id = get_jwt_identity()
        
        notification = Notification.query.filter_by(
            id=notification_id,
            recipient_id=user_id
        ).first()
        
        if not notification:
            return jsonify({'message': 'Notification not found'}), 404
        
        notification.is_read = True
        db.session.commit()
        
        return jsonify({
            'message': 'Notification marked as read',
            'notification': notification.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error marking notification as read: {str(e)}'}), 500

@notifications_bp.route('/mark-all-read', methods=['PUT'])
@jwt_required()
def mark_all_notifications_read():
    try:
        from models import Notification
        
        user_id = get_jwt_identity()
        
        Notification.query.filter_by(
            recipient_id=user_id,
            is_read=False
        ).update({'is_read': True})
        
        db.session.commit()
        
        return jsonify({'message': 'All notifications marked as read'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error marking notifications as read: {str(e)}'}), 500

@notifications_bp.route('/unread-count', methods=['GET'])
@jwt_required()
def get_unread_count():
    try:
        from models import Notification
        
        user_id = get_jwt_identity()
        
        count = Notification.query.filter_by(
            recipient_id=user_id,
            is_read=False
        ).count()
        
        return jsonify({'unread_count': count}), 200
        
    except Exception as e:
        return jsonify({'message': f'Error getting unread count: {str(e)}'}), 500

@notifications_bp.route('/delete/<int:notification_id>', methods=['DELETE'])
@jwt_required()
def delete_notification(notification_id):
    try:
        from models import Notification
        
        user_id = get_jwt_identity()
        
        notification = Notification.query.filter_by(
            id=notification_id,
            recipient_id=user_id
        ).first()
        
        if not notification:
            return jsonify({'message': 'Notification not found'}), 404
        
        db.session.delete(notification)
        db.session.commit()
        
        return jsonify({'message': 'Notification deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error deleting notification: {str(e)}'}), 500
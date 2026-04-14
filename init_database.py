"""
Initialize Database for Food Share System
Run this script to create all required tables
"""

import sqlite3
import os
from datetime import datetime

def initialize_database():
    """Create all database tables"""
    try:
        db_path = 'solapur_food_share.db'
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                phone VARCHAR(15) UNIQUE NOT NULL,
                user_type VARCHAR(20) NOT NULL,
                taluka VARCHAR(50),
                area VARCHAR(100),
                address TEXT,
                is_verified BOOLEAN DEFAULT 0,
                is_admin BOOLEAN DEFAULT 0,
                organization_name VARCHAR(200),
                registration_number VARCHAR(50),
                capacity INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create foods table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                donor_id INTEGER NOT NULL,
                title VARCHAR(200) NOT NULL,
                description TEXT NOT NULL,
                food_type VARCHAR(50) NOT NULL,
                quantity VARCHAR(100) NOT NULL,
                container_type VARCHAR(50) NOT NULL,
                expiry_time DATETIME NOT NULL,
                pickup_address TEXT NOT NULL,
                taluka VARCHAR(50),
                area VARCHAR(100),
                gps_latitude REAL,
                gps_longitude REAL,
                images TEXT,
                status VARCHAR(20) DEFAULT 'available',
                claimed_by INTEGER,
                claimed_at DATETIME,
                qr_code VARCHAR(255) UNIQUE,
                pickup_code VARCHAR(10) UNIQUE,
                contact_person_name VARCHAR(100),
                contact_person_phone VARCHAR(15),
                special_instructions TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (donor_id) REFERENCES users (id),
                FOREIGN KEY (claimed_by) REFERENCES users (id)
            )
        ''')
        
        # Create notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient_id INTEGER NOT NULL,
                sender_id INTEGER NOT NULL,
                food_id INTEGER NOT NULL,
                type VARCHAR(50) NOT NULL,
                title VARCHAR(200) NOT NULL,
                message TEXT NOT NULL,
                taluka VARCHAR(50),
                area VARCHAR(100),
                is_read BOOLEAN DEFAULT 0,
                sent_via_sms BOOLEAN DEFAULT 0,
                sent_via_app BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recipient_id) REFERENCES users (id),
                FOREIGN KEY (sender_id) REFERENCES users (id),
                FOREIGN KEY (food_id) REFERENCES foods (id)
            )
        ''')
        
        # Create email_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ngo_email VARCHAR(120) NOT NULL,
                food_id INTEGER NOT NULL,
                sent_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'pending',
                error_message TEXT,
                FOREIGN KEY (food_id) REFERENCES foods (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("Database initialized successfully!")
        print("Tables created: users, foods, notifications, email_logs")
        
    except Exception as e:
        print(f"Database initialization failed: {str(e)}")

def create_test_ngos():
    """Create test NGOs for testing"""
    try:
        conn = sqlite3.connect('solapur_food_share.db')
        cursor = conn.cursor()
        
        # Test NGOs data
        test_ngos = [
            {
                'name': 'Annapurna NGO',
                'email': 'annapurna@gmail.com',
                'phone': '9876543210',
                'user_type': 'ngo',
                'taluka': 'Solapur City',
                'area': 'Solapur City',
                'address': 'Solapur City',
                'organization_name': 'Annapurna Food Bank',
                'is_verified': 1
            },
            {
                'name': 'Seva Foundation',
                'email': 'seva@gmail.com',
                'phone': '9876543211',
                'user_type': 'ngo',
                'taluka': 'Pandharpur',
                'area': 'Pandharpur',
                'address': 'Pandharpur',
                'organization_name': 'Seva Foundation',
                'is_verified': 1
            },
            {
                'name': 'Food Relief Center',
                'email': 'relief@gmail.com',
                'phone': '9876543212',
                'user_type': 'ngo',
                'taluka': 'Barshi',
                'area': 'Barshi',
                'address': 'Barshi',
                'organization_name': 'Food Relief Center',
                'is_verified': 1
            }
        ]
        
        # Insert test NGOs
        for ngo in test_ngos:
            # Check if NGO already exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (ngo['email'],))
            if cursor.fetchone() is None:
                cursor.execute('''
                    INSERT INTO users (name, email, phone, user_type, taluka, area, address, 
                                     organization_name, is_verified, password_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ngo['name'], ngo['email'], ngo['phone'], ngo['user_type'],
                    ngo['taluka'], ngo['area'], ngo['address'], ngo['organization_name'],
                    ngo['is_verified'], 'test_password_hash'
                ))
                print(f"Created NGO: {ngo['name']} in {ngo['taluka']}")
            else:
                print(f"NGO already exists: {ngo['email']}")
        
        conn.commit()
        conn.close()
        
        print("Test NGOs created successfully!")
        
    except Exception as e:
        print(f"Failed to create test NGOs: {str(e)}")

if __name__ == "__main__":
    print("=== DATABASE INITIALIZATION ===")
    print()
    
    print("1. Creating database tables...")
    initialize_database()
    print()
    
    print("2. Creating test NGOs...")
    create_test_ngos()
    print()
    
    print("=== INITIALIZATION COMPLETE ===")
    print()
    print("Now you can:")
    print("1. Test email notifications")
    print("2. Submit food donations")
    print("3. Check email logs")
    print("4. Verify NGO targeting")

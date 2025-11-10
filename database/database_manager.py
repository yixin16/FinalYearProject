# database/database_manager.py
import sqlite3
import logging
import numpy as np
from datetime import datetime
import csv
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from models.cnn_face_model import extract_features
from flask import flash, g, current_app
marked_students = set()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('attendance.db')
        g.db.row_factory = sqlite3.Row
        # Enable foreign key constraints
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db

def initialize_database():
    """Initialize the database with the enhanced schema"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    with open('schema.sql', 'r') as f:
        cursor.executescript(f.read())
    
    conn.commit()
    logger.info("Database initialized with enhanced schema")
    return conn

# ========== User Management ==========
def create_user(username, password, email, role):
    """Create a new user with hashed password"""
    try:
        db = get_db()
        hashed_pw = generate_password_hash(password)
        db.execute(
            "INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
            (username, hashed_pw, email, role)
        )
        db.commit()
        return db.execute("SELECT last_insert_rowid()").fetchone()[0]
    except sqlite3.IntegrityError as e:
        logger.error(f"User creation failed: {e}")
        return None

def validate_login(username, password):
    """Validate user credentials with password hashing"""
    db = get_db()
    user = db.execute(
        "SELECT id, username, password, role FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    
    if user and check_password_hash(user['password'], password):
        return user
    return None


def create_course(course_code, course_name, lecturer_id, description=None):
    """Create a new course with proper validation"""
    db = get_db()
    
    try:
        # Verify the lecturer exists and is valid
        lecturer = db.execute(
            """SELECT 1 FROM users
            WHERE id = ? AND role = 'lecturer'""",  # Added id check
            (lecturer_id,)
        ).fetchone()
        
        if not lecturer:
            logger.error(f"Invalid lecturer ID or not a lecturer: {lecturer_id}")
            return False
            
        # Create the course
        db.execute(
            """INSERT INTO courses 
            (course_code, course_name, description, created_by) 
            VALUES (?, ?, ?, ?)""",
            (course_code, course_name, description, lecturer_id)
        )
        db.commit()
        return True
        
    except sqlite3.IntegrityError as e:
        logger.error(f"Course creation failed: {e}")
        db.rollback()
        return False
    

# ========== Attendance Session Management ==========
def start_attendance_session(db, section_id, lecturer_id):
    """Start a new attendance session"""
    try:
        # Check for existing active session
        existing = db.execute(
            "SELECT id FROM attendance_sessions WHERE class_section_id = ? AND status = 'active'",
            (section_id,)
        ).fetchone()
        
        if existing:
            return False
            
        db.execute(
            """INSERT INTO attendance_sessions 
            (class_section_id, lecturer_id, start_time, status)
            VALUES (?, ?, datetime('now'), 'active')""",
            (section_id, lecturer_id)
        )
        db.commit()
        return True
    except sqlite3.Error:
        return False

def end_attendance_session(db, section_id):
    """End the current active session"""
    try:
        result = db.execute(
            """UPDATE attendance_sessions 
            SET end_time = datetime('now'), status = 'completed'
            WHERE class_section_id = ? AND status = 'active'""",
            (section_id,)
        )
        db.commit()
        return result.rowcount > 0
    except sqlite3.Error:
        return False

def record_manual_attendance(db, section_id, student_id, status='present'):
    """Record manual attendance"""
    try:
        # Get current active session
        session = db.execute(
            "SELECT id FROM attendance_sessions WHERE class_section_id = ? AND status = 'active'",
            (section_id,)
        ).fetchone()
        
        if not session:
            flash("No active session to record against", "warning")
            return False
            
        # Check if already recorded
        existing = db.execute(
            "SELECT 1 FROM attendance_records WHERE session_id = ? AND student_id = ?",
            (session['id'], student_id)
        ).fetchone()
        
        if existing:
            flash("Attendance already recorded for this student", "warning")
            return False
            
        db.execute(
            """INSERT INTO attendance_records
            (session_id, student_id, status, method, timestamp)
            VALUES (?, ?, ?, 'manual', datetime('now'))""",
            (session['id'], student_id, status)
        )
        db.commit()
        flash("Attendance recorded successfully", "success")
        return True
    except sqlite3.Error as e:
        flash(f"Failed to record attendance: {str(e)}", "danger")
        return False

def generate_secure_token(session_id, section_id):
    """Generate a secure token for QR codes"""
    import hashlib
    secret = current_app.config['SECRET_KEY']
    data = f"{session_id}-{section_id}-{datetime.now().timestamp()}"
    return hashlib.sha256(f"{data}{secret}".encode()).hexdigest()


def fetch_stored_embeddings():
    """Fetch all face embeddings from the database"""
    print("In fetch_stored_embeddings =")
    db = get_db()
    try:
        rows = db.execute(
            'SELECT id, name, embedding FROM students WHERE embedding IS NOT NULL'
        ).fetchall()
        
        embeddings = {}
        for row in rows:
            embeddings[row['name']] = (row['id'], np.frombuffer(row['embedding'], dtype=np.float32))
        return embeddings
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        return {}


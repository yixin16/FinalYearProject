from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify, Response
import base64
import cv2
import sqlite3
from mtcnn import MTCNN
from models.cnn_face_model import load_face_model, extract_features
from database.database_manager import *
from datetime import datetime, timedelta
import numpy as np
import time
from collections import deque
from utils.image_processing import preprocess_image
from utils.face_recognition import recognize_face
from utils.liveness_detection import * 
from utils.download_face_images import *
from utils.report import *
from flask import send_file
import pandas as pd
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Required for session management
app.config['DATABASE'] = 'attendance.db'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
from datetime import datetime

try:
    current_time = datetime.now()
    print("Success! The current time is:", current_time)
except AttributeError as e:
    print("The error is still happening, even in a clean file.")
    print(e)
# Initialize face detection and recognition models
face_model = load_face_model()

# Database connection management
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
        # Enable foreign key constraints
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.executescript(f.read())
            
def validate_manual_entry(form_data):
    """Validate manual attendance entry form data"""
    required = ['student_id']
    return all(field in form_data and form_data[field] for field in required)

def get_current_session(db, section_id):
    """Get current active session for a section"""
    return db.execute(
        "SELECT * FROM attendance_sessions WHERE class_section_id = ? AND status = 'active'",
        (section_id,)
    ).fetchone()
      
# ========== Authentication Routes ==========
@app.route("/", methods=["GET", "POST"])
def index():
    if 'user_id' in session:
        return redirect(url_for("select_role"))
    return redirect(url_for("login"))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        # Here you would typically:
        # 1. Check if email exists in your database
        # 2. Generate a reset token
        # 3. Send password reset email
        # 4. Show success message
        
        flash('If an account exists with this email, a password reset link has been sent', 'info')
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        role = request.form['role']
        if create_user(username, password, email, role):
            flash("Registration successful. Please login.")
            return redirect(url_for("login"))
        else:
            flash("Registration failed. Please try again.")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user = validate_login(username, password)
        if user:
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['username'] = user['username']
            flash("Login successful", "success")
            return redirect(url_for("select_role"))
        else:
            flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ========== Dashboard Routes ==========
@app.route("/select_role")
def select_role():
    if 'user_id' not in session:
        return redirect(url_for("login"))
    if session['role'] == 'lecturer':
        return redirect(url_for("lecturer_dashboard"))
    elif session['role'] == 'student':
        return redirect(url_for("student_dashboard"))
    else:
        flash("Invalid role.", "danger")
        return redirect(url_for("login"))
       
@app.route('/student_dashboard')
def student_dashboard():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))

    db = get_db()
    
    try:
        # Get student information
        student = db.execute(
            """SELECT s.id, s.face_registered, u.username, u.email 
               FROM students s JOIN users u ON s.user_id = u.id 
               WHERE s.user_id = ?""",
            (session['user_id'],)
        ).fetchone()
        
        if not student:
            flash("Student record not found", "danger")
            return redirect(url_for('login'))
            
        # Get enrolled courses
        courses = db.execute("""
            SELECT c.id, c.course_code, c.course_name
            FROM enrollments e
            JOIN courses c ON e.course_id = c.id
            WHERE e.student_id = ?
            ORDER BY c.course_name
        """, (student['id'],)).fetchall()

        # Get sections for each course
        course_sections = {}
        for course in courses:
            sections = db.execute("""
                SELECT cs.id, cs.section_name, cs.schedule,
                       EXISTS (
                           SELECT 1 FROM attendance_records ar
                           WHERE ar.student_id = ? AND ar.session_id = cs.id
                       ) AS has_marked
                FROM class_sections cs
                WHERE cs.course_id = ?
                ORDER BY cs.section_name
            """, (student['id'], course['id'])).fetchall()

            course_sections[course['id']] = sections

        return render_template('student_dashboard.html',
                            student=student,
                            courses=courses,
                            course_sections=course_sections,
                            face_registered=student['face_registered'])
        
    except Exception as e:
        flash(f"Error loading dashboard: {str(e)}", "danger")
        return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))

    mode = request.args.get('mode', 'scan')
    student_id = session['user_id']
    initial_challenge = ""

    if mode == 'scan':
        try:
            db = get_db()
            challenge_row = db.execute(
                "SELECT challenge FROM liveness_data WHERE student_id = ? AND timestamp > ?",
                (student_id, time.time() - 30)
            ).fetchone()
            initial_challenge = challenge_row['challenge'] if challenge_row else ""
            close_db(None)
        except Exception as e:
            app.logger.error(f"Challenge fetch error: {str(e)}")

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_liveness_status')
def get_liveness_status():
    """Endpoint to check liveness status for the current user"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401

    student_id = session['user_id']
    
    try:
        db = get_db()
        # Get the most recent liveness data (within last 30 seconds)
        liveness_data = db.execute(
            "SELECT liveness_score, liveness_reason FROM liveness_data "
            "WHERE student_id = ? AND timestamp > ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (student_id, time.time() - 30)
        ).fetchone()
        
        if liveness_data:
            is_live = liveness_data['liveness_score'] > 60  # Threshold you defined in your video_feed
            return jsonify({
                'success': True,
                'is_live': is_live,
                'score': liveness_data['liveness_score'],
                'message': liveness_data['liveness_reason'] if not is_live else "Liveness verified"
            })
        else:
            return jsonify({
                'success': False,
                'is_live': False,
                'message': "No recent liveness data found"
            })
            
    except Exception as e:
        logger.error(f"Error getting liveness status: {str(e)}")
        return jsonify({
            'success': False,
            'is_live': False,
            'message': "Error checking liveness status"
        }), 500
    finally:
        close_db(None)
        
@app.route('/register_face', methods=['POST'])
def register_face():
    """Face registration endpoint that stores images in filesystem only"""
    if 'user_id' not in session or session.get('role') != 'student':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    try:
        db = get_db()
        # Get basic student info
        student = db.execute(
            "SELECT id, name, face_registered FROM students WHERE user_id = ?",
            (session['user_id'],)
        ).fetchone()

        if not student:
            return jsonify({'success': False, 'message': 'Student record not found'}), 404

        # Check image upload
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        # Read and validate image
        image_data = image_file.read()
        if len(image_data) > 5 * 1024 * 1024:  # 5MB max
            return jsonify({'success': False, 'message': 'Image too large (max 5MB)'}), 400

        # Process image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        detector = MTCNN()
        faces = detector.detect_faces(img)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'}), 400
            
        if len(faces) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected'}), 400

        # Get best face
        face = max(faces, key=lambda f: f['confidence'])
        if face['confidence'] < 0.9:
            return jsonify({
                'success': False, 
                'message': f'Low confidence ({face["confidence"]:.2f})'
            }), 400

        # Extract face with padding
        x, y, w, h = face['box']
        padding = int(max(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2*padding, img.shape[1] - x)
        h = min(h + 2*padding, img.shape[0] - y)
        face_img = img[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return jsonify({'success': False, 'message': 'Failed to extract face'}), 400

        # Save face image
        success, current_count = save_face_image(
            face_img, 
            student['name'], 
            student['id']
        )
        
        if not success:
            return jsonify({'success': False, 'message': 'Failed to save image'}), 500

        # Generate preview
        _, buffer = cv2.imencode('.jpg', cv2.resize(face_img, (150, 150)))
        face_preview = base64.b64encode(buffer).decode('utf-8')

        # Complete registration if we have enough images
        registration_complete = (current_count >= 5)
        if registration_complete:
            processed_face = preprocess_image(face_img)
            embedding = extract_features(processed_face)
            if embedding is None:
                return jsonify({'success': False, 'message': 'Failed to generate face embedding'}), 400
        
            # Convert embedding to BLOB and update in the database
            embedding_blob = embedding.tobytes()
            db.execute(
                "UPDATE students SET embedding = ?, face_registered = 1 WHERE id = ?",
                (embedding_blob, student['id'],)
            )
            db.commit()

        return jsonify({
            'success': True,
            'message': 'Face captured successfully',
            'face_preview': face_preview,
            'images_captured': current_count,
            'registration_complete': registration_complete,
            'progress': f"{current_count}/5 images captured"
        })

    except Exception as e:
        current_app.logger.error(f"Face registration error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False, 
            'message': 'An error occurred during registration'
        }), 500
    
def align_face(rgb_img, landmarks):
    """Align face using dlib landmarks while preserving RGB color."""
    try:
        # Left and right eye coordinates
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center between eyes
        eyes_center_x = (left_eye[0] + right_eye[0]) // 2
        eyes_center_y = (left_eye[1] + right_eye[1]) // 2
        
        # --- FIX: Explicitly cast coordinates to native Python int() ---
        # This prevents the "wrong type" error from OpenCV.
        eyes_center = (int(eyes_center_x), int(eyes_center_y))
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        
        # Rotate RGB image
        (w, h) = (rgb_img.shape[1], rgb_img.shape[0])
        rotated = cv2.warpAffine(rgb_img, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Crop the face from the rotated image
        eye_dist = np.sqrt(dx**2 + dy**2)
        crop_size = int(eye_dist * 2.2) # A multiplier of 2.2 usually gives good padding
        
        x = int(eyes_center[0] - crop_size // 2)
        y = int(eyes_center[1] - crop_size // 2)
        
        # Ensure coordinates are within the bounds of the rotated image
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + crop_size)
        y2 = min(h, y + crop_size)
        
        cropped = rotated[y1:y2, x1:x2]
        if cropped.size == 0:
            return None
            
        # Return a standard size for your CNN, e.g., 160x160 for FaceNet
        return cv2.resize(cropped, (160, 160)) 

    except Exception as e:
        current_app.logger.error(f"Error during face alignment: {e}")
        return None
        
@app.route('/scan_attendance/<int:section_id>', methods=['POST'])
def scan_attendance(section_id):
    """Enhanced attendance scanning with liveness detection and comprehensive validation"""
    # Authentication and authorization check
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    db = None
    try:
        db = get_db()
        
        # ========== VALIDATION PHASE ==========
        # 1. Verify student enrollment in section
        enrollment_check = db.execute(
            """SELECT s.id, s.embedding, cs.id as section_id,
                    cs.active, cs.section_name, c.course_code, c.id as course_id
            FROM students s
            JOIN enrollments e ON s.id = e.student_id
            JOIN courses c ON e.course_id = c.id
            JOIN class_sections cs ON c.id = cs.course_id
            WHERE s.user_id = ? AND cs.id = ?""",
            (session['user_id'], section_id)
        ).fetchone()

        if not enrollment_check:
            return jsonify({
                'success': False, 
                'message': 'Not enrolled in this section'
            }), 403

        if not enrollment_check['embedding']:
            return jsonify({
                'success': False,
                'message': 'Complete face registration first'
            }), 400

        if not enrollment_check['active']:
            return jsonify({
                'success': False,
                'message': 'This class session is not active'
            }), 400

        # 2. Check for existing attendance
        existing_attendance = db.execute(
            """SELECT id FROM attendance_records 
               WHERE student_id = ? AND session_id = ?""",
            (enrollment_check['id'], enrollment_check['section_id'])
        ).fetchone()

        if existing_attendance:
            return jsonify({
                'success': False,
                'message': 'Attendance already marked for this session'
            }), 400

        # ========== IMAGE PROCESSING PHASE ==========
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided'
            }), 400

        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No selected file'
            }), 400

        # Validate image file
        try:
            img_bytes = file.read()
            if len(img_bytes) == 0:
                return jsonify({
                    'success': False,
                    'message': 'Empty image file'
                }), 400

            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({
                    'success': False,
                    'message': 'Invalid image format'
                }), 400
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Error processing image'
            }), 400

        # ========== LIVENESS DETECTION PHASE ==========
        try:
            # Convert to grayscale for dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using dlib
            faces = dlib_face_detector(gray, 0)
            
            if not faces:
                return jsonify({
                    'success': False,
                    'message': 'No face detected in the image'
                }), 400

            # Get the first face
            face = faces[0]
            
            # Get facial landmarks
            shape = dlib_predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            # # Perform liveness detection
            # liveness_detector = EnhancedLivenessDetector()
            # # Reset state without keep_challenge parameter
            # if hasattr(liveness_detector, 'reset_state'):
            #     liveness_detector.reset_state()  # Call without parameter
            # analysis = liveness_detector.analyze_frame(frame, landmarks)
            
            # if not analysis['is_live'] or analysis['confidence'] < 0.8:
            #     return jsonify({
            #         'success': False,
            #         'message': 'Liveness verification failed',
            #         'instruction': analysis['instruction'],
            #         'confidence': float(analysis['confidence'])
            #     }), 400

        except Exception as e:
            logger.error(f"Liveness detection error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Error during liveness verification'
            }), 500

        # ========== FACE VERIFICATION PHASE ==========
        try:
            # Extract face region using dlib coordinates
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                return jsonify({
                    'success': False,
                    'message': 'Invalid face region detected'
                }), 400

            # Preprocess and extract features
            preprocessed_face = preprocess_image(face_crop)
            if preprocessed_face is None:
                return jsonify({
                    'success': False,
                    'message': 'Face preprocessing failed'
                }), 400

            current_embedding = extract_features(preprocessed_face)
            if current_embedding is None:
                return jsonify({
                    'success': False,
                    'message': 'Feature extraction failed'
                }), 400

            # Get stored embeddings (consider caching this)
            stored_embeddings = fetch_stored_embeddings()
            if not stored_embeddings:
                return jsonify({
                    'success': False,
                    'message': 'No registered faces found'
                }), 500

            # Perform recognition
            student_id, student_name, similarity = recognize_face(
                current_embedding, 
                stored_embeddings,
                threshold=0.90,
                adaptive_threshold=True
            )

            if not student_id:
                return jsonify({
                    'success': False,
                    'message': f'Verification failed (score: {similarity:.2f})',
                    'similarity': float(similarity)
                }), 403
        except Exception as e:
            logger.error(f"Face recognition error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Error during face verification'
            }), 500

        # ========== ATTENDANCE RECORDING PHASE ==========
        try:
            with db:  
                db.execute(
                    """INSERT INTO attendance_records 
                       (student_id, session_id, status, confidence_score, 
                        timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (enrollment_check['id'], 
                     enrollment_check['section_id'], 
                     'present', 
                     float(similarity),
                     datetime.now().isoformat())
                )

            logger.info(f"Attendance recorded for student {enrollment_check['id']} in section {section_id}")
            
            return jsonify({
                'success': True,
                'student_id': enrollment_check['id'],
                'similarity': float(similarity),
                'course': enrollment_check['course_code'],
                'section': enrollment_check['section_name'],
                'timestamp': datetime.now().isoformat()
            })
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Database error recording attendance'
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in scan_attendance: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500
    finally:
        if db:
            db.close()
                     
@app.route('/update_liveness', methods=['POST'])
def update_liveness():
    """
    Efficient liveness tracking endpoint that:
    - Updates liveness metadata without storing images
    - Maintains security and validation
    - Uses minimal database storage
    """
    # Authentication check
    if 'user_id' not in session:
        return jsonify({
            'success': False,
            'message': 'Authentication required',
            'code': 'AUTH_REQUIRED'
        }), 401

    student_id = session['user_id']
    db = None
    
    try:
        # Get liveness score from request (0-100)
        try:
            data = request.get_json()
            liveness_score = float(data.get('score', 0))
            challenge = data.get('challenge', '')
        except Exception as e:
            return jsonify({
                'success': False,
                'message': 'Invalid request data',
                'code': 'INVALID_DATA'
            }), 400

        # Validate score range
        if not 0 <= liveness_score <= 100:
            return jsonify({
                'success': False,
                'message': 'Invalid liveness score',
                'code': 'INVALID_SCORE'
            }), 400

        # Database operations
        db = get_db()
        try:
            # Verify student exists
            student_exists = db.execute(
                "SELECT 1 FROM students WHERE user_id = ?",
                (student_id,)
            ).fetchone()
            
            if not student_exists:
                return jsonify({
                    'success': False,
                    'message': 'Student record not found',
                    'code': 'STUDENT_NOT_FOUND'
                }), 404

            # Update liveness metadata
            db.execute(
                """INSERT OR REPLACE INTO liveness_data 
                   (student_id, score, challenge, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (student_id, liveness_score, challenge, time.time())
            )
            
            db.commit()

            return jsonify({
                'success': True,
                'message': 'Liveness updated',
                'timestamp': time.time()
            })

        except sqlite3.Error as e:
            db.rollback()
            logger.error(f"Database error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Database error',
                'code': 'DB_ERROR'
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

    finally:
        if db:
            close_db(None)

@app.route("/lecturer_dashboard")
def lecturer_dashboard():
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for("login"))

    db = get_db()
    
    # Get basic stats - fixed query
    stats = db.execute("""
        SELECT
            COUNT(DISTINCT c.id) as course_count,
            COUNT(DISTINCT e.student_id) as student_count,
            COUNT(DISTINCT CASE WHEN date(ar.timestamp) = date('now') THEN ar.id END) as today_attendance,
            0 as pending_actions
        FROM courses c
        LEFT JOIN enrollments e ON c.id = e.course_id
        LEFT JOIN class_sections cs ON c.id = cs.course_id
        LEFT JOIN attendance_sessions sess ON cs.id = sess.class_section_id
        LEFT JOIN attendance_records ar ON sess.id = ar.session_id
        WHERE c.created_by = ?
    """, (session['user_id'],)).fetchone()

    # Get courses with their sections
    courses = db.execute("""
        SELECT c.id, c.course_code, c.course_name
        FROM courses c
        WHERE c.created_by = ?
    """, (session['user_id'],)).fetchall()

    courses_with_sections = []
    for course in courses:
        sections = db.execute("""
    SELECT cs.id, cs.section_name, cs.schedule, cs.location,
        COUNT(e.student_id) as student_count
    FROM class_sections cs
    LEFT JOIN enrollments e ON cs.course_id = e.course_id
    WHERE cs.course_id = ?
    GROUP BY cs.id
""", (course['id'],)).fetchall()
        courses_with_sections.append({
            **course,
            'sections': sections
        })

    return render_template("lecturer_dashboard.html",
                         course_count=stats['course_count'],
                         student_count=stats['student_count'],
                         today_attendance=stats['today_attendance'],
                         pending_actions=stats['pending_actions'],
                         courses_with_sections=courses_with_sections)
    
@app.route("/section/<int:section_id>")
def view_section(section_id):
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for("login"))

    db = get_db()

    # Get section + course info
    section = db.execute("""
        SELECT cs.*, c.course_name, c.course_code
        FROM class_sections cs
        JOIN courses c ON cs.course_id = c.id
        WHERE cs.id = ?
    """, (section_id,)).fetchone()

    if not section:
        flash("Section not found", "danger")
        return redirect(url_for("lecturer_dashboard"))

    # Count enrolled students (course-level enrollment)
    enrolled_count = db.execute("""
        SELECT COUNT(*) AS total
        FROM enrollments e
        JOIN users u ON u.id = e.student_id
        WHERE e.course_id = ?
    """, (section["course_id"],)).fetchone()["total"]

    # Count unique attendance dates for this section
    session_count = db.execute("""
        SELECT COUNT(DISTINCT timestamp) AS total
        FROM attendance_records
        WHERE session_id = ?
    """, (section_id,)).fetchone()["total"]

    # Recent attendance (last 5 dates, section-specific)
    recent_sessions = db.execute("""
        SELECT timestamp,
               SUM(CASE WHEN status='present' THEN 1 ELSE 0 END) AS present_count,
               SUM(CASE WHEN status='absent' THEN 1 ELSE 0 END) AS absent_count,
               SUM(CASE WHEN status='late' THEN 1 ELSE 0 END) AS late_count
        FROM attendance_records
        WHERE session_id = ?
        GROUP BY timestamp
        ORDER BY timestamp DESC
        LIMIT 5
    """, (section_id,)).fetchall()

    # Student attendance summary (course enrollment + section attendance)
    student_summary = db.execute("""
        SELECT u.username,
               ROUND(AVG(CASE WHEN ar.status='present' THEN 1.0
                              WHEN ar.status='late' THEN 0.5
                              ELSE 0 END) * 100, 1) AS attendance_rate,
               (SELECT status
                FROM attendance_records ar2
                WHERE ar2.student_id = u.id AND ar2.session_id = ?
                ORDER BY ar2.timestamp DESC
                LIMIT 1) AS last_status
        FROM users u
        JOIN enrollments e ON u.id = e.student_id
        LEFT JOIN attendance_records ar
               ON u.id = ar.student_id AND ar.session_id = ?
        WHERE e.course_id = ?
        GROUP BY u.id
    """, (section_id, section_id, section["course_id"])).fetchall()

    return render_template(
        "view_section.html",
        section=section,
        enrolled_count=enrolled_count,
        session_count=session_count,
        recent_sessions=recent_sessions,
        student_summary=student_summary
    )

@app.route("/courses", methods=["GET", "POST"])
def courses():
    if "user_id" not in session or session["role"] != "lecturer":
        return redirect(url_for("login"))

    db = get_db()

    if request.method == "POST":
        course_code = request.form["course_code"]
        course_name = request.form["course_name"]
        lecturer_id = session["user_id"]
        description = request.form.get("description", "")

        if create_course(course_code, course_name, lecturer_id, description):
            flash("Course created successfully", "success")
        else:
            flash("Course creation failed (course code may already exist)", "danger")
        return redirect(url_for("courses"))

    # Get all courses created by this lecturer
    courses = db.execute("""
        SELECT c.id, c.course_code, c.course_name, c.description,
               COUNT(cs.id) AS section_count
        FROM courses c
        LEFT JOIN class_sections cs ON c.id = cs.course_id
        WHERE c.created_by = ?
        GROUP BY c.id
        ORDER BY c.course_name
    """, (session["user_id"],)).fetchall()

    # Attach sections to each course
    course_list = []
    for course in courses:
        sections = db.execute("""
            SELECT cs.id, cs.section_name, cs.schedule, cs.location,
                   COUNT(e.student_id) AS student_count
            FROM class_sections cs
            LEFT JOIN enrollments e ON cs.id = e.course_id
            WHERE cs.course_id = ?
            GROUP BY cs.id
            ORDER BY cs.section_name
        """, (course["id"],)).fetchall()

        course_dict = dict(course)
        course_dict["sections"] = [dict(s) for s in sections]
        course_list.append(course_dict)

    return render_template("courses.html", courses=course_list)

@app.route("/course/<int:course_id>/sections", methods=["GET", "POST"])
def manage_sections(course_id):
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for("login"))
    
    db = get_db()
    
    try:
        # Get course details with lecturer info
        course = db.execute("""
            SELECT c.*, u.username as lecturer_name 
            FROM courses c
            JOIN users u ON c.created_by = u.id
            WHERE c.id = ?
        """, (course_id,)).fetchone()
        
        if not course:
            flash("Course not found", "danger")
            return redirect(url_for("lecturer_dashboard"))

        if request.method == "POST":
            section_name = request.form.get("section_name")
            schedule = request.form.get("schedule")
            location = request.form.get("location", "")
            
            if not section_name or not schedule:
                flash("Section name and schedule are required", "danger")
                return redirect(url_for("manage_sections", course_id=course_id))
            
            try:
                db.execute("""
                    INSERT INTO class_sections 
                    (course_id, section_name, schedule, location, active)
                    VALUES (?, ?, ?, ?, 1)
                """, (course_id, section_name, schedule, location))
                db.commit()
                flash("Class section created successfully", "success")
            except sqlite3.IntegrityError:
                db.rollback()
                flash("Section creation failed (section name may already exist)", "danger")
            
            return redirect(url_for("manage_sections", course_id=course_id))
        
        # Get sections with session counts (modified query)
        sections = db.execute("""
            SELECT cs.*, 
                   (SELECT COUNT(*) FROM attendance_sessions 
                    WHERE class_section_id = cs.id) as session_count
            FROM class_sections cs
            WHERE cs.course_id = ?
            ORDER BY cs.section_name
        """, (course_id,)).fetchall()
        
        return render_template("class_sections.html", 
                             course=course, 
                             sections=sections)
        
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for("lecturer_dashboard"))

@app.route('/enroll_student', methods=['GET', 'POST'])
def enroll_student():
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for('login'))

    db = get_db()

    if request.method == 'POST':
        student_user_id = request.form.get('student_id')
        course_id = request.form.get('course_id')

        try:
            # Get student record
            student = db.execute(
                "SELECT id FROM students WHERE user_id = ?", 
                (student_user_id,)
            ).fetchone()

            if not student:
                flash("Student record not found", "danger")
                return redirect(url_for('enroll_student'))

            student_id = student['id']

            # Verify course belongs to this lecturer
            course_exists = db.execute(
                "SELECT 1 FROM courses WHERE id = ? AND created_by = ?",
                (course_id, session['user_id'])
            ).fetchone()

            if not course_exists:
                flash("Unauthorized: Course does not exist or not owned by you", "danger")
                return redirect(url_for('enroll_student'))

            # Check if already enrolled
            already_enrolled = db.execute(
                "SELECT 1 FROM enrollments WHERE student_id = ? AND course_id = ?",
                (student_id, course_id)
            ).fetchone()
            
            if already_enrolled:
                flash('Student is already enrolled in this course', 'warning')
            else:
                db.execute(
                    "INSERT INTO enrollments (student_id, course_id) VALUES (?, ?)",
                    (student_id, course_id)
                )
                db.commit()
                flash('Student enrolled in course successfully', 'success')

        except sqlite3.IntegrityError as e:
            db.rollback()
            flash(f'Database error: {str(e)}', 'danger')
        except Exception as e:
            db.rollback()
            flash(f'Unexpected error: {str(e)}', 'danger')

        return redirect(url_for('enroll_student'))
    
    # 🔹 GET request - better data retrieval
    students = db.execute("""
        SELECT 
            u.id AS user_id, 
            u.username, 
            s.id AS student_id,
            COALESCE(s.name, u.username) AS name
        FROM users u
        JOIN students s ON u.id = s.user_id
        ORDER BY s.name
    """).fetchall()
    
    courses = db.execute("""
        SELECT 
            c.id, 
            c.course_code, 
            c.course_name,
            COUNT(cs.id) AS section_count
        FROM courses c
        LEFT JOIN class_sections cs ON c.id = cs.course_id
        WHERE c.created_by = ?
        GROUP BY c.id, c.course_code, c.course_name
        ORDER BY c.course_code
    """, (session['user_id'],)).fetchall()
    
    return render_template('enroll_student.html', students=students, courses=courses)

@app.route("/course/<int:course_id>/edit", methods=["GET", "POST"])
def edit_course(course_id):
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for('login'))

    db = get_db()
    course = db.execute("""
        SELECT *
        FROM courses
        WHERE id = ? AND created_by = ?
    """, (course_id, session['user_id'])).fetchone()

    if not course:
        flash("Course not found or access denied", "danger")
        return redirect(url_for('lecturer_dashboard'))

    if request.method == "POST":
        course_name = request.form.get('course_name', '').strip()
        course_code = request.form.get('course_code', '').strip()
        description = request.form.get('description', '').strip()

        if not course_name or not course_code:
            flash("Course name and course code are required.", "danger")
            return render_template("edit_course.html", course=course)

        try:
            db.execute("""
                UPDATE courses
                SET course_name = ?, course_code = ?, description = ?
                WHERE id = ?
            """, (course_name, course_code, description, course_id))
            db.commit()
            flash("Course updated successfully", "success")
            return redirect(url_for('courses'))
        except Exception as e:
            db.rollback()
            flash(f"Database error: {str(e)}", "danger")

    return render_template("edit_course.html", course=course)

@app.route("/course/<int:course_id>/delete")
def delete_course(course_id):
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for('login'))

    db = get_db()

    # Verify course belongs to lecturer
    course = db.execute("""
        SELECT * FROM courses
        WHERE id = ? AND created_by = ?
    """, (course_id, session['user_id'])).fetchone()

    if not course:
        flash("Course not found or access denied.", "danger")
        return redirect(url_for("courses"))

    try:
        # Delete attendance records tied to sections in this course
        db.execute("""
            DELETE FROM attendance_records
            WHERE session_id IN (
                SELECT id FROM class_sections WHERE course_id = ?
            )
        """, (course_id,))

        # Delete class sections under the course
        db.execute("DELETE FROM class_sections WHERE course_id = ?", (course_id,))

        # Delete enrollments tied to this course
        db.execute("DELETE FROM enrollments WHERE course_id = ?", (course_id,))

        # Finally delete the course itself
        db.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        db.commit()

        flash(f"Course '{course['course_code']} - {course['course_name']}' deleted successfully.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error deleting course: {str(e)}", "danger")

    return redirect(url_for("courses"))

@app.route("/section/<int:section_id>/attendance", methods=["GET", "POST"])
def manage_attendance(section_id):
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for("login"))
    
    db = get_db()
    qr_data = None
    
    try:
        # ✅ Fix: enrollments uses course_id, not class_section_id
        section = db.execute("""
            SELECT cs.*, c.course_code, c.course_name,
                   COUNT(e.student_id) as enrolled_count
            FROM class_sections cs
            JOIN courses c ON cs.course_id = c.id
            LEFT JOIN enrollments e ON c.id = e.course_id
            WHERE cs.id = ? AND c.created_by = ?
            GROUP BY cs.id
        """, (section_id, session['user_id'])).fetchone()
        
        if not section:
            flash("Class section not found or access denied", "danger")
            return redirect(url_for("lecturer_dashboard"))  # ✅ FIX endpoint

        if request.method == "POST":
            action = request.form.get("action")
            
            if action == "start_session":
                if not start_attendance_session(db, section_id, session['user_id']):
                    flash("Failed to start session or session already exists", "danger")
                else:
                    session_info = db.execute("""
                        SELECT * FROM attendance_sessions 
                        WHERE class_section_id = ? AND status = 'active'
                    """, (section_id,)).fetchone()
                    
                    qr_data = {
                        'session_id': session_info['id'],
                        'section_id': section_id,
                        'expires': (datetime.now() + timedelta(minutes=15)).isoformat(),
                        'token': generate_secure_token(session_info['id'], section_id)
                    }
                    flash("Attendance session started", "success")

            elif action == "end_session":
                if end_attendance_session(db, section_id):
                    flash("Session ended successfully", "success")
                else:
                    flash("No active session to end", "warning")

            elif action == "manual_entry":
                student_id = request.form.get("student_id")
                status = request.form.get("status", "present")
                
                if not student_id:
                    flash("Please select a student", "danger")
                else:
                    record_manual_attendance(db, section_id, student_id, status)
            
            return redirect(url_for('manage_attendance', section_id=section_id))

        # Get current session data
        current_session = db.execute("""
            SELECT sess.*, COUNT(ar.id) as record_count
            FROM attendance_sessions sess
            LEFT JOIN attendance_records ar ON sess.id = ar.session_id
            WHERE sess.class_section_id = ? AND sess.status = 'active'
            GROUP BY sess.id
        """, (section_id,)).fetchone()

        # ✅ Fix: enrolled students via course_id
        enrolled_students = db.execute("""
            SELECT s.id, s.name
            FROM students s
            JOIN enrollments e ON s.id = e.student_id
            JOIN class_sections cs ON e.course_id = cs.course_id
            WHERE cs.id = ?
            ORDER BY s.name
        """, (section_id,)).fetchall()

        # Attendance records (paginated)
        page = request.args.get('page', 1, type=int)
        per_page = 20
        offset = (page - 1) * per_page
        
        attendance_records = db.execute("""
            SELECT s.name, ar.timestamp, ar.status
            FROM attendance_records ar
            JOIN students s ON ar.student_id = s.id
            JOIN attendance_sessions sess ON ar.session_id = sess.id
            WHERE sess.class_section_id = ?
            ORDER BY ar.timestamp DESC
            LIMIT ? OFFSET ?
        """, (section_id, per_page, offset)).fetchall()

        total_records = db.execute("""
            SELECT COUNT(*) 
            FROM attendance_records ar
            JOIN attendance_sessions sess ON ar.session_id = sess.id
            WHERE sess.class_section_id = ?
        """, (section_id,)).fetchone()[0]

        # Attendance statistics
        attendance_stats = db.execute("""
            SELECT 
                COUNT(DISTINCT sess.id) as total_sessions,
                COUNT(DISTINCT e.student_id) as total_students,
                COUNT(DISTINCT ar.student_id) as attended_students,
                ROUND(100.0 * COUNT(DISTINCT ar.student_id) / NULLIF(COUNT(DISTINCT e.student_id), 0), 1) as attendance_rate
            FROM class_sections cs
            LEFT JOIN attendance_sessions sess ON cs.id = sess.class_section_id
            LEFT JOIN attendance_records ar ON sess.id = ar.session_id
            JOIN enrollments e ON cs.course_id = e.course_id
            WHERE cs.id = ?
        """, (section_id,)).fetchone()

        return render_template("manage_attendance.html",
                               section=section,
                               current_session=current_session,
                               enrolled_students=enrolled_students,
                               attendance_records=attendance_records,
                               attendance_stats=attendance_stats,
                               qr_data=qr_data,
                               pagination={
                                   'page': page,
                                   'per_page': per_page,
                                   'total': total_records,
                                   'pages': (total_records + per_page - 1) // per_page
                               })

    except sqlite3.Error as e:
        flash(f"Database error: {str(e)}", "danger")
        return redirect(url_for("lecturer_dashboard"))  # ✅ FIX endpoint

@app.route("/section/<int:section_id>/edit", methods=["GET", "POST"])
def edit_section(section_id):
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for('login'))

    db = get_db()
    section = db.execute("""
        SELECT cs.*, c.course_name, c.course_code
        FROM class_sections cs
        JOIN courses c ON cs.course_id = c.id
        WHERE cs.id = ? AND c.created_by = ?
    """, (section_id, session['user_id'])).fetchone()
    
    if not section:
        flash("Section not found or access denied", "danger")
        return redirect(url_for('lecturer_dashboard'))

    if request.method == "POST":
        section_name = request.form.get('section_name', '').strip()
        schedule = request.form.get('schedule', '').strip()
        location = request.form.get('location', '').strip()

        if not section_name or not schedule:
            flash("Section name and schedule are required.", "danger")
            return render_template("edit_section.html", section=section)

        try:
            db.execute("""
                UPDATE class_sections
                SET section_name = ?, schedule = ?, location = ?
                WHERE id = ?
            """, (section_name, schedule, location, section_id))
            db.commit()
            flash("Section updated successfully", "success")
            return redirect(url_for('view_section', section_id=section_id))
        except Exception as e:
            db.rollback()
            flash(f"Database error: {str(e)}", "danger")

    return render_template("edit_section.html", section=section)

@app.route("/section/<int:section_id>/delete")
def delete_section(section_id):
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for("login"))

    db = get_db()

    # Verify section belongs to a course created by this lecturer
    section = db.execute("""
        SELECT cs.*, c.course_name, c.course_code
        FROM class_sections cs
        JOIN courses c ON cs.course_id = c.id
        WHERE cs.id = ? AND c.created_by = ?
    """, (section_id, session['user_id'])).fetchone()

    if not section:
        flash("Section not found or access denied.", "danger")
        return redirect(url_for("lecturer_dashboard"))

    try:
        # Delete attendance tied to this section
        db.execute("DELETE FROM attendance_records WHERE session_id = ?", (section_id,))

        # Delete the section
        db.execute("DELETE FROM class_sections WHERE id = ?", (section_id,))
        db.commit()

        flash(f"Section '{section['section_name']}' from {section['course_code']} deleted successfully.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error deleting section: {str(e)}", "danger")

    return redirect(url_for("manage_sections", course_id=section["course_id"]))

@app.route('/export_reports', methods=['GET', 'POST'])
def export_reports():
    if 'user_id' not in session or session['role'] != 'lecturer':
        return redirect(url_for('login'))

    db = get_db()

    try:
        # Get courses and sections taught by this lecturer
        courses_with_sections = db.execute("""
            SELECT c.id AS course_id, c.course_code, c.course_name,
                   cs.id AS section_id, cs.section_name
            FROM courses c
            JOIN class_sections cs ON c.id = cs.course_id
            WHERE c.created_by = ?
            ORDER BY c.course_code, cs.section_name
        """, (session['user_id'],)).fetchall()

        if request.method == 'POST':
            section_id = request.form.get('section_id')
            export_format = request.form.get('format')

            if not section_id:
                flash('Please select a class section', 'danger')
                return redirect(url_for('export_reports'))

            # Verify lecturer owns this section
            section_info = db.execute("""
                SELECT c.course_code, c.course_name, cs.section_name
                FROM class_sections cs
                JOIN courses c ON cs.course_id = c.id
                WHERE cs.id = ? AND c.created_by = ?
            """, (section_id, session['user_id'])).fetchone()

            if not section_info:
                flash('Unauthorized access to section data', 'danger')
                return redirect(url_for('export_reports'))

            # ✅ Retrieve attendance only from attendance_records for students enrolled in this section
            records = db.execute("""
                SELECT
                    s.id AS student_id,
                    s.name AS student_name,
                    ar.timestamp,
                    ar.status,
                    ar.confidence_score
                FROM attendance_records ar
                JOIN students s ON ar.student_id = s.id
                WHERE ar.student_id IN (
                    SELECT student_id FROM enrollments
                    WHERE course_id = (SELECT course_id FROM class_sections WHERE id = ?)
                )
                ORDER BY s.name, ar.timestamp
            """, (section_id,)).fetchall()

            if not records:
                flash('No attendance records found for this section.', 'warning')
                return redirect(url_for('export_reports'))

            # Convert to DataFrame
            df_raw = pd.DataFrame(records, columns=[
                'Student ID', 'Name', 'Timestamp', 'Status', 'Confidence Score'
            ])

            df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'])
            df_raw['Session Date'] = df_raw['Timestamp'].dt.strftime('%Y-%m-%d')
            df_raw['Timestamp'] = df_raw['Timestamp'].dt.strftime('%H:%M:%S')
            df_raw['Confidence Score'] = df_raw['Confidence Score'].apply(lambda x: f"{x:.1%}" if x else 'N/A')

            # Pivot table for summary
            df_pivot = df_raw.pivot_table(
                index=['Student ID', 'Name'],
                columns='Session Date',
                values='Status',
                aggfunc='first',
                fill_value='Absent'
            ).reset_index()

            # Export based on selected format
            if export_format == 'csv':
                output = BytesIO()
                df_pivot.to_csv(output, index=False)
                output.seek(0)
                filename = f"attendance_{section_info['course_code']}_{section_info['section_name']}.csv".replace(' ', '_')
                return send_file(output, mimetype='text/csv', as_attachment=True, download_name=filename)

            elif export_format == 'excel':
                return create_excel_report(df_raw, df_pivot, section_info)

            elif export_format == 'pdf':
                return create_pdf_report(df_raw, df_pivot, section_info)

    except Exception as e:
        print(f"ERROR in /export_reports: {e}")
        flash('An unexpected error occurred while generating the report.', 'danger')
        return redirect(url_for('export_reports'))

    # Prepare courses for dropdown
    courses = {}
    for row in courses_with_sections:
        if row['course_id'] not in courses:
            courses[row['course_id']] = {
                'course_code': row['course_code'],
                'course_name': row['course_name'],
                'sections': []
            }
        courses[row['course_id']]['sections'].append({
            'section_id': row['section_id'],
            'section_name': row['section_name']
        })

    return render_template('export_reports.html', courses=courses.values())

# ========== Main Application ==========
if __name__ == "__main__":
    init_db()
    app.run(debug=True, threaded=True)
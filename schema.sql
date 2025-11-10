CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('lecturer', 'student')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- Separate primary key
    user_id INTEGER NOT NULL UNIQUE, -- Foreign key to users
    name TEXT NOT NULL,
    face_registered BOOLEAN DEFAULT 0,
    embedding BLOB,
    FOREIGN KEY (user_id) REFERENCES users (id) ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS lecturers (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL UNIQUE,
    course_name TEXT NOT NULL,
    description TEXT,
    created_by INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS class_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL,
    section_name TEXT NOT NULL,
    schedule TEXT NOT NULL,  -- e.g., "Mon/Wed 10:00-11:30"
    location TEXT,
    active BOOLEAN DEFAULT 0,
    FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE,
    UNIQUE (course_id, section_name)
);


CREATE TABLE IF NOT EXISTS enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,  -- Now links directly to course
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    UNIQUE (student_id, course_id)  -- Prevent duplicate enrollments
);

CREATE TABLE IF NOT EXISTS attendance_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_section_id INTEGER NOT NULL,
    lecturer_id INTEGER NOT NULL,
    session_date DATE NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    qr_code_token TEXT UNIQUE,
    qr_code_expiry TIMESTAMP,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'completed')),
    FOREIGN KEY (class_section_id) REFERENCES class_sections (id) ON DELETE CASCADE,
    FOREIGN KEY (lecturer_id) REFERENCES lecturers (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'present' CHECK (status IN ('present', 'absent', 'late', 'excused')),
    confidence_score REAL, 
    liveness_score REAL,
    FOREIGN KEY (session_id) REFERENCES class_sections (id),
    FOREIGN KEY (student_id) REFERENCES students (id),
    UNIQUE (session_id, student_id)  -- Prevent duplicate attendance per session
);

CREATE TABLE IF NOT EXISTS attendance_exceptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    class_section_id INTEGER NOT NULL,
    session_id INTEGER,
    exception_type TEXT NOT NULL CHECK (exception_type IN ('early', 'late', 'other')),
    reason TEXT,
    approved BOOLEAN DEFAULT 0,
    approved_by INTEGER,
    approved_at TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
    FOREIGN KEY (class_section_id) REFERENCES class_sections (id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES attendance_sessions (id) ON DELETE SET NULL,
    FOREIGN KEY (approved_by) REFERENCES lecturers (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS liveness_data (
    student_id INTEGER PRIMARY KEY,
    liveness_score FLOAT NOT NULL DEFAULT 0,
    liveness_reason TEXT NOT NULL DEFAULT '',
    challenge TEXT NOT NULL DEFAULT '',
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id)
);

CREATE TRIGGER IF NOT EXISTS after_user_insert_student
AFTER INSERT ON users
WHEN NEW.role = 'student'
BEGIN
    INSERT INTO students (user_id, name)
    VALUES (NEW.id, NEW.username); -- or whatever column contains the student's name
END;

CREATE TRIGGER IF NOT EXISTS after_user_insert_lecturer
AFTER INSERT ON users
WHEN NEW.role = 'lecturer'
BEGIN
    INSERT INTO lecturers (user_id, name)
    VALUES (NEW.id, NEW.username); -- or whatever column contains the lecturer's name
END;

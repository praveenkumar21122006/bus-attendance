import os
import io
import sqlite3
import pickle
import base64
import numpy as np
import cv2
from functools import wraps
from datetime import datetime, date

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    Response,
    send_from_directory,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "bus-attendance-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.environ.get("DATA_DIR", BASE_DIR)
os.makedirs(STORAGE_DIR, exist_ok=True)
DATABASE = os.path.join(STORAGE_DIR, "database.db")
DATASET_DIR = os.path.join(STORAGE_DIR, "dataset")
CAPTURES_DIR = os.path.join(STORAGE_DIR, "captures")
ENCODINGS_PATH = os.path.join(STORAGE_DIR, "face_encodings.pkl")

# Live recognition state
live_capture = None
live_student_name = None
mark_done = False


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            bus_no TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            date_of_birth TEXT NOT NULL DEFAULT '',
            password TEXT NOT NULL DEFAULT '',
            boarding_point TEXT NOT NULL DEFAULT '',
            fees_amount REAL NOT NULL DEFAULT 0,
            year_of_study TEXT NOT NULL DEFAULT '',
            bus_no TEXT NOT NULL,
            face_encoding BLOB,
            image_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            bus_no TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'present',
            image_path TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );

        CREATE TABLE IF NOT EXISTS unknown_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            bus_no TEXT NOT NULL DEFAULT 'Unknown',
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unknown',
            image_path TEXT
        );
        """
    )

    attendance_cols = conn.execute("PRAGMA table_info(attendance)").fetchall()
    attendance_col_names = {col[1] for col in attendance_cols}
    if "image_path" not in attendance_col_names:
        try:
            conn.execute("ALTER TABLE attendance ADD COLUMN image_path TEXT")
        except sqlite3.OperationalError:
            pass

    admin_cols = {col[1] for col in conn.execute("PRAGMA table_info(admin)").fetchall()}
    if "bus_no" not in admin_cols:
        conn.execute("ALTER TABLE admin ADD COLUMN bus_no TEXT NOT NULL DEFAULT ''")

    student_cols = {col[1] for col in conn.execute("PRAGMA table_info(students)").fetchall()}
    if "password" not in student_cols:
        conn.execute("ALTER TABLE students ADD COLUMN password TEXT NOT NULL DEFAULT ''")
        conn.execute("UPDATE students SET password = roll_no WHERE password = ''")
    for column, definition in (
        ("date_of_birth", "TEXT NOT NULL DEFAULT ''"),
        ("boarding_point", "TEXT NOT NULL DEFAULT ''"),
        ("fees_amount", "REAL NOT NULL DEFAULT 0"),
        ("year_of_study", "TEXT NOT NULL DEFAULT ''"),
    ):
        if column not in student_cols:
            conn.execute(f"ALTER TABLE students ADD COLUMN {column} {definition}")

    unknown_cols = conn.execute("PRAGMA table_info(unknown_attendance)").fetchall()
    unknown_col_names = {col[1] for col in unknown_cols}
    if "image_path" not in unknown_col_names:
        try:
            conn.execute("ALTER TABLE unknown_attendance ADD COLUMN image_path TEXT")
        except sqlite3.OperationalError:
            pass

    cur = conn.execute("SELECT COUNT(*) AS c FROM admin")
    if cur.fetchone()["c"] == 0:
        conn.execute(
            "INSERT INTO admin (username, password, bus_no) VALUES (?, ?, ?)",
            ("admin", "admin123", ""),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Face encoding helpers
# ---------------------------------------------------------------------------
def load_encodings(bus_no=None):
    encodings = {}
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                encodings = pickle.load(f)
        except (EOFError, ValueError, pickle.PickleError):
            encodings = {}
    if bus_no:
        encodings = {}

    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, roll_no, face_encoding FROM students WHERE face_encoding IS NOT NULL"
        + (" AND bus_no = ?" if bus_no else ""),
        (bus_no,) if bus_no else (),
    ).fetchall()
    conn.close()

    for row in rows:
        try:
            encoding = pickle.loads(row["face_encoding"])
        except Exception:
            continue
        encodings[row["roll_no"]] = {
            "id": row["id"],
            "name": row["name"],
            "roll_no": row["roll_no"],
            "encoding": encoding,
        }
    return encodings


def save_encodings(encodings):
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)


def find_matching_student(image, bus_no=None):
    try:
        import face_recognition
    except ImportError:
        return None

    encodings = load_encodings(bus_no)
    if not encodings:
        return None

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        unknown_encs = face_recognition.face_encodings(rgb)
        if not unknown_encs:
            return None

        known_names = []
        known_encs = []
        for student in encodings.values():
            known_names.append(student["name"])
            known_encs.append(student["encoding"])

        matches = face_recognition.compare_faces(known_encs, unknown_encs[0])
        if not any(matches):
            return None

        distances = face_recognition.face_distance(known_encs, unknown_encs[0])
        best_index = int(np.argmin(distances))
        if matches[best_index] and distances[best_index] < 0.5:
            return known_names[best_index]
    except Exception:
        return None

    return None


def current_attendance_session():
    return "all_day"


def get_face_encodings(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        from face_recognition import face_encodings, face_locations
    except ImportError:
        return []
    locations = face_locations(rgb)
    return face_encodings(rgb, locations)


# ---------------------------------------------------------------------------
# Login decorators
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


def master_admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if session.get("admin_bus_no"):
            flash("Only the master admin can manage bus logins.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)

    return decorated


def student_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("student_logged_in"):
            return redirect(url_for("student_login"))
        return f(*args, **kwargs)

    return decorated


def attendance_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not (session.get("admin_logged_in") or session.get("student_logged_in")):
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if session.get("admin_logged_in"):
        return redirect(url_for("dashboard"))
    if session.get("student_logged_in"):
        return redirect(url_for("live_attendance"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        conn = get_db()
        row = conn.execute(
            "SELECT * FROM admin WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        conn.close()
        if row:
            session.clear()
            session["admin_logged_in"] = True
            session["admin_username"] = username
            session["admin_bus_no"] = row["bus_no"] or None
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/add_admin", methods=["GET", "POST"])
@master_admin_required
def add_admin():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        bus_no = request.form.get("bus_no", "").strip()
        if not username or not password or not bus_no:
            flash("Username, password, and bus number are required.", "error")
            return redirect(url_for("add_admin"))

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO admin (username, password, bus_no) VALUES (?, ?, ?)",
                (username, password, bus_no),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            flash("That admin username already exists.", "error")
            return redirect(url_for("add_admin"))
        conn.close()
        flash(f"Admin login for bus {bus_no} created.", "success")
        return redirect(url_for("add_admin"))

    conn = get_db()
    admins = conn.execute(
        "SELECT username, bus_no FROM admin WHERE bus_no != '' ORDER BY bus_no, username"
    ).fetchall()
    conn.close()
    return render_template("add_admin.html", admins=admins)


@app.route("/student-login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip()
        password = request.form.get("password", "")
        conn = get_db()
        student = conn.execute(
            "SELECT id, name, roll_no FROM students WHERE roll_no = ? AND password = ?",
            (roll_no, password),
        ).fetchone()
        conn.close()
        if student:
            session.clear()
            session["student_logged_in"] = True
            session["student_id"] = student["id"]
            session["student_name"] = student["name"]
            session["student_roll_no"] = student["roll_no"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("live_attendance"))
        flash("Invalid roll number or password.", "error")
    return render_template("student_login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    bus_no = session.get("admin_bus_no")
    scope_sql = " WHERE bus_no = ?" if bus_no else ""
    scope_params = (bus_no,) if bus_no else ()
    students = conn.execute(
        f"SELECT COUNT(*) AS c FROM students{scope_sql}", scope_params
    ).fetchone()["c"]
    today = date.today().isoformat()
    present_today = conn.execute(
        "SELECT COUNT(DISTINCT student_id) AS c FROM attendance WHERE date = ? AND status = 'present'"
        + (" AND bus_no = ?" if bus_no else ""),
        (today, bus_no) if bus_no else (today,),
    ).fetchone()["c"]
    absent_students = conn.execute(
        """SELECT id, name, roll_no, bus_no
           FROM students
              WHERE NOT EXISTS (
               SELECT 1 FROM attendance
               WHERE attendance.student_id = students.id
                 AND attendance.date = ?
                                 AND attendance.status = 'present'
           )
              """ + (" AND students.bus_no = ?" if bus_no else "") + """
           ORDER BY bus_no ASC, name ASC""",
          (today, bus_no) if bus_no else (today,),
    ).fetchall()
    recent = conn.execute(
        """SELECT attendance.date, attendance.time, students.name, students.roll_no,
                   attendance.bus_no, attendance.status
            FROM attendance
            JOIN students ON students.id = attendance.student_id
            """ + ("WHERE attendance.bus_no = ?" if bus_no else "") + """
            ORDER BY attendance.id DESC LIMIT 10"""
        , (bus_no,) if bus_no else ()
    ).fetchall()
    conn.close()
    return render_template(
        "dashboard.html",
        total_students=students,
        present_today=present_today,
        absent_students=absent_students,
        recent=recent,
    )


@app.route("/add_student", methods=["GET", "POST"])
@login_required
def add_student():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        date_of_birth = request.form.get("date_of_birth", "").strip()
        password = request.form.get("password", "").strip() or roll_no
        boarding_point = request.form.get("boarding_point", "").strip()
        fees_amount = request.form.get("fees_amount", "").strip()
        year_of_study = request.form.get("year_of_study", "").strip()
        bus_no = request.form.get("bus_no", "").strip()
        if session.get("admin_bus_no"):
            bus_no = session["admin_bus_no"]
        image_data = request.form.get("image_data", "")
        id_card_file = request.files.get("id_card")

        if not (name and roll_no and date_of_birth and boarding_point and fees_amount and year_of_study and bus_no):
            flash("All fields are required.", "error")
            return redirect(url_for("add_student"))

        try:
            fees_amount = float(fees_amount)
            if fees_amount < 0:
                raise ValueError
        except ValueError:
            flash("Fees amount must be a valid non-negative number.", "error")
            return redirect(url_for("add_student"))

        conn = get_db()
        existing = conn.execute(
            "SELECT id, name, roll_no, boarding_point, bus_no FROM students WHERE roll_no = ?"
            + (" AND bus_no = ?" if session.get("admin_bus_no") else ""),
            (roll_no, session["admin_bus_no"]) if session.get("admin_bus_no") else (roll_no,),
        ).fetchone()
        if existing:
            conn.close()
            return render_template(
                "add_student.html",
                admin_bus_no=session.get("admin_bus_no"),
                existing_student=existing,
            )

        image_path = None
        try:
            from face_recognition import face_encodings as fr_encodings
        except ImportError:
            fr_encodings = None

        if (image_data or id_card_file) and fr_encodings:
            try:
                if id_card_file and id_card_file.filename:
                    img_bytes = id_card_file.read()
                else:
                    img_bytes = base64.b64decode(image_data.split(",")[1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("The ID card image could not be read")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = fr_encodings(rgb)
                if not encs:
                    flash("No face detected on the ID card. Upload a clear card image or capture a face photo.", "error")
                    conn.close()
                    return redirect(url_for("add_student"))

                # Save the ID card image as the student's registered evidence.
                student_dir = os.path.join(DATASET_DIR, roll_no)
                os.makedirs(student_dir, exist_ok=True)
                image_path = os.path.join(student_dir, f"{roll_no}_id_card.jpg")
                cv2.imwrite(image_path, img)

                # Save encoding for all registered faces
                encodings = load_encodings()
                encodings[roll_no] = {
                    "name": name,
                    "roll_no": roll_no,
                    "encoding": encs[0],
                }
                save_encodings(encodings)

                # Persist the binary encoding (pickle)
                enc_bytes = pickle.dumps(encs[0])
            except Exception as e:
                conn.close()
                flash(f"Failed to process face image: {e}", "error")
                return redirect(url_for("add_student"))
        else:
            enc_bytes = None

        conn.execute(
            "INSERT INTO students "
            "(name, roll_no, date_of_birth, password, boarding_point, fees_amount, year_of_study, bus_no, face_encoding, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, roll_no, date_of_birth, password, boarding_point, fees_amount, year_of_study, bus_no, enc_bytes, image_path),
        )
        conn.commit()
        conn.close()
        flash(f"Student {name} added successfully.", "success")
        return redirect(url_for("add_student"))

    return render_template("add_student.html", admin_bus_no=session.get("admin_bus_no"))


@app.route("/students")
@login_required
def students():
    conn = get_db()
    bus_no = session.get("admin_bus_no")
    rows = conn.execute(
        "SELECT * FROM students" + (" WHERE bus_no = ?" if bus_no else "") + " ORDER BY bus_no ASC, name ASC",
        (bus_no,) if bus_no else (),
    ).fetchall()
    conn.close()
    students_by_bus = {}
    for row in rows:
        students_by_bus.setdefault(row["bus_no"], []).append(row)
    ordered_buses = sorted(students_by_bus.keys())
    return render_template(
        "attendance.html",
        students_by_bus=students_by_bus,
        ordered_buses=ordered_buses,
        mode="students",
    )


@app.route("/delete_student/<int:student_id>", methods=["POST"])
@login_required
def delete_student(student_id):
    conn = get_db()
    bus_no = session.get("admin_bus_no")
    student = conn.execute(
        "SELECT id, name, roll_no, image_path, face_encoding FROM students WHERE id = ?"
        + (" AND bus_no = ?" if bus_no else ""),
        (student_id, bus_no) if bus_no else (student_id,),
    ).fetchone()

    if student:
        if student["image_path"] and os.path.exists(student["image_path"]):
            try:
                os.remove(student["image_path"])
            except OSError:
                pass

        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()

        encodings = load_encodings()
        encodings.pop(student["roll_no"], None)
        save_encodings(encodings)

        flash(f"Student {student['name']} deleted successfully.", "success")
    else:
        flash("Student not found.", "error")

    conn.close()
    return redirect(url_for("students"))

@app.route("/delete_attendance/<int:attendance_id>", methods=["POST"])
@login_required
def delete_attendance(attendance_id):
    conn = get_db()
    bus_no = session.get("admin_bus_no")
    deleted = conn.execute(
        "DELETE FROM attendance WHERE id = ?" + (" AND bus_no = ?" if bus_no else ""),
        (attendance_id, bus_no) if bus_no else (attendance_id,),
    ).rowcount
    conn.commit()
    conn.close()
    flash(
        "Attendance log deleted." if deleted else "Attendance log not found.",
        "success" if deleted else "error",
    )
    return redirect(url_for("attendance"))


@app.route("/delete_unknown_attendance/<int:record_id>", methods=["POST"])
@login_required
def delete_unknown_attendance(record_id):
    conn = get_db()
    bus_no = session.get("admin_bus_no")
    record = conn.execute(
        "SELECT image_path FROM unknown_attendance WHERE id = ?"
        + (" AND bus_no = ?" if bus_no else ""),
        (record_id, bus_no) if bus_no else (record_id,),
    ).fetchone()
    if record:
        conn.execute(
            "DELETE FROM unknown_attendance WHERE id = ?"
            + (" AND bus_no = ?" if bus_no else ""),
            (record_id, bus_no) if bus_no else (record_id,),
        )
        conn.commit()
    conn.close()

    if record and record["image_path"]:
        image_path = os.path.join(STORAGE_DIR, record["image_path"])
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

    flash(
        "Unknown-person log deleted." if record else "Unknown-person log not found.",
        "success" if record else "error",
    )
    return redirect(url_for("unknown_attendance"))


@app.route("/live_attendance")
@attendance_login_required
def live_attendance():
    return render_template("live_attendance.html", student=None)


@app.route("/verify_attendance", methods=["POST"])
@attendance_login_required
def verify_attendance():
    roll_no = request.form.get("roll_no", "").strip()
    date_of_birth = request.form.get("date_of_birth", "").strip()
    if not roll_no or not date_of_birth:
        flash("Register number and date of birth are required.", "error")
        return redirect(url_for("live_attendance"))

    conn = get_db()
    bus_no = session.get("admin_bus_no")
    student = conn.execute(
        "SELECT id, name, roll_no, date_of_birth, bus_no FROM students WHERE roll_no = ? AND date_of_birth = ?"
        + (" AND bus_no = ?" if bus_no else ""),
        (roll_no, date_of_birth, bus_no) if bus_no else (roll_no, date_of_birth),
    ).fetchone()
    conn.close()
    if not student:
        flash("The register number and date of birth do not match.", "error")
        return redirect(url_for("live_attendance"))
    if session.get("student_logged_in") and student["id"] != session.get("student_id"):
        flash("You can only mark attendance for your own account.", "error")
        return redirect(url_for("live_attendance"))

    return render_template("mark_attendance.html", student=student)


@app.route("/mark_verified_attendance", methods=["POST"])
@attendance_login_required
def mark_verified_attendance():
    student_id = request.form.get("student_id", type=int)
    status = request.form.get("status", "").lower()
    if status not in {"present", "absent"} or not student_id:
        flash("Choose Present or Absent to continue.", "error")
        return redirect(url_for("live_attendance"))

    conn = get_db()
    bus_no = session.get("admin_bus_no")
    student = conn.execute(
        "SELECT id, name, roll_no, bus_no FROM students WHERE id = ?"
        + (" AND bus_no = ?" if bus_no else ""),
        (student_id, bus_no) if bus_no else (student_id,),
    ).fetchone()
    if not student or (session.get("student_logged_in") and student["id"] != session.get("student_id")):
        conn.close()
        flash("Student verification failed.", "error")
        return redirect(url_for("live_attendance"))

    today = date.today().isoformat()
    existing = conn.execute(
        "SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id, today)
    ).fetchone()
    if existing:
        conn.close()
        flash(f"Attendance is already marked for {student['name']} today.", "error")
        return redirect(url_for("live_attendance"))

    conn.execute(
        "INSERT INTO attendance (student_id, bus_no, date, time, status) VALUES (?, ?, ?, ?, ?)",
        (student["id"], student["bus_no"], today, datetime.now().strftime("%H:%M:%S"), status),
    )
    conn.commit()
    conn.close()
    flash(f"{student['name']} marked {status}.", "success")
    return redirect(url_for("live_attendance"))


@app.route("/attendance")
@login_required
def attendance():
    conn = get_db()
    bus_no = session.get("admin_bus_no")
    attendance_cols = {col[1] for col in conn.execute("PRAGMA table_info(attendance)").fetchall()}
    known_sql = """
        SELECT attendance.id, attendance.date, attendance.time, attendance.bus_no,
               attendance.status, students.name, students.roll_no,
               students.year_of_study, students.boarding_point
        FROM attendance
        JOIN students ON students.id = attendance.student_id
        """ + ("WHERE attendance.bus_no = ?" if bus_no else "") + """
        ORDER BY attendance.id DESC
    """
    if "image_path" in attendance_cols:
        known_sql = """
            SELECT attendance.id, attendance.date, attendance.time, attendance.bus_no,
                   attendance.status, COALESCE(attendance.image_path, students.image_path) AS image_path,
                       students.name, students.roll_no, students.year_of_study,
                       students.boarding_point
            FROM attendance
            JOIN students ON students.id = attendance.student_id
            """ + ("WHERE attendance.bus_no = ?" if bus_no else "") + """
            ORDER BY attendance.id DESC
        """

    known_rows = conn.execute(known_sql, (bus_no,) if bus_no else ()).fetchall()
    conn.close()

    records = []
    for row in known_rows:
        image_path = row["image_path"] if "image_path" in row.keys() else None
        if image_path and os.path.isabs(image_path):
            image_path = os.path.relpath(image_path, STORAGE_DIR)
        records.append({
            "id": row["id"],
            "date": row["date"],
            "time": row["time"],
            "bus_no": row["bus_no"],
            "status": row["status"],
            "name": row["name"],
            "roll_no": row["roll_no"],
            "year_of_study": row["year_of_study"],
            "boarding_point": row["boarding_point"],
            "image_path": image_path,
            "kind": "known",
        })
    records.sort(key=lambda item: (item["date"], item["time"]), reverse=True)
    return render_template("attendance.html", records=records, mode="attendance")


# ---------------------------------------------------------------------------
# Live recognition (video + capture callback)
# ---------------------------------------------------------------------------
def gen_frames(bus_no=None):
    global live_capture, live_student_name, mark_done
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return
    try:
        from face_recognition import compare_faces, face_distance, face_locations, face_encodings
    except ImportError:
        compare_faces = None

    mark_done = False
    live_student_name = None
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        current_name = None
        if compare_faces:
            encodings = load_encodings(bus_no)
            known = list(encodings.keys())
            if known:
                known_encs = [encodings[k]["encoding"] for k in known]
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                boxes = face_locations(rgb)

                for (top, right, bottom, left) in boxes:
                    enc = face_encodings(rgb, [(top, right, bottom, left)])
                    if not enc:
                        continue
                    matches = compare_faces(known_encs, enc[0])
                    dists = face_distance(known_encs, enc[0])
                    if len(dists) > 0:
                        best = int(np.argmin(dists))
                        if matches[best] and dists[best] < 0.5:
                            current_name = encodings[known[best]]["name"]
                    top, right, bottom, left = (
                        top * 2,
                        right * 2,
                        bottom * 2,
                        left * 2,
                    )
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        current_name or "Unknown",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                if current_name:
                    live_student_name = current_name
                else:
                    live_student_name = None

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            break
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
    camera.release()


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(
        gen_frames(session.get("admin_bus_no")),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/mark_attendance", methods=["POST"])
@attendance_login_required
def mark_attendance():
    global live_student_name, mark_done

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image_data") or request.form.get("image_data") or ""
    student_name = None
    captured_image_path = None

    if image_data:
        try:
            img_bytes = base64.b64decode(image_data.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                student_name = find_matching_student(img, session.get("admin_bus_no"))
                if student_name:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    capture_dir = os.path.join(CAPTURES_DIR, "recognized")
                    os.makedirs(capture_dir, exist_ok=True)
                    capture_path = os.path.join(capture_dir, f"{student_name}_{timestamp}.jpg")
                    if cv2.imwrite(capture_path, img):
                        captured_image_path = os.path.relpath(capture_path, STORAGE_DIR)
        except Exception:
            student_name = None
    elif live_student_name and not mark_done:
        student_name = live_student_name

    if student_name:
        conn = get_db()
        bus_no = session.get("admin_bus_no")
        student = conn.execute(
            "SELECT * FROM students WHERE name = ?"
            + (" AND bus_no = ?" if bus_no else ""),
            (student_name, bus_no) if bus_no else (student_name,),
        ).fetchone()
        if student:
            if session.get("student_logged_in") and student["id"] != session.get("student_id"):
                return jsonify({"status": "waiting"})
            today = date.today().isoformat()
            now = datetime.now().strftime("%H:%M:%S")
            existing = conn.execute(
                "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
                (student["id"], today),
            ).fetchone()
            if existing:
                conn.close()
                return jsonify({"status": "already", "name": student_name})
            image_path = captured_image_path or (os.path.relpath(student["image_path"], STORAGE_DIR) if student["image_path"] else None)
            conn.execute(
                "INSERT INTO attendance (student_id, bus_no, date, time, image_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (student["id"], student["bus_no"], today, now, image_path),
            )
            conn.commit()
            conn.close()
            mark_done = True
            live_student_name = student_name
            return jsonify({"status": "ok", "name": student_name})
        conn.close()

    return jsonify({"status": "waiting"})


@app.route("/reset_live", methods=["POST"])
@login_required
def reset_live():
    global mark_done, live_student_name
    mark_done = False
    live_student_name = None
    return jsonify({"ok": True})


@app.route("/media/<path:filename>")
@login_required
def media_file(filename):
    return send_from_directory(STORAGE_DIR, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
else:
    init_db()

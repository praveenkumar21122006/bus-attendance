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
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "bus-attendance-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "database.db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
ENCODINGS_PATH = os.path.join(BASE_DIR, "face_encodings.pkl")

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
            password TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
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
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
        """
    )
    # Seed default admin (admin / admin123)
    cur = conn.execute("SELECT COUNT(*) AS c FROM admin")
    if cur.fetchone()["c"] == 0:
        conn.execute(
            "INSERT INTO admin (username, password) VALUES (?, ?)",
            ("admin", "admin123"),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Face encoding helpers
# ---------------------------------------------------------------------------
def load_encodings():
    encodings = {}
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                encodings = pickle.load(f)
        except (EOFError, ValueError, pickle.PickleError):
            encodings = {}

    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, roll_no, face_encoding FROM students WHERE face_encoding IS NOT NULL"
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


def find_matching_student(image):
    try:
        import face_recognition
    except ImportError:
        return None

    encodings = load_encodings()
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if session.get("admin_logged_in"):
        return redirect(url_for("dashboard"))
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
            session["admin_logged_in"] = True
            session["admin_username"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    students = conn.execute("SELECT COUNT(*) AS c FROM students").fetchone()["c"]
    today = date.today().isoformat()
    present_today = conn.execute(
        "SELECT COUNT(DISTINCT student_id) AS c FROM attendance WHERE date = ?",
        (today,),
    ).fetchone()["c"]
    recent = conn.execute(
        """SELECT attendance.date, attendance.time, students.name, students.roll_no,
                   attendance.bus_no, attendance.status
            FROM attendance
            JOIN students ON students.id = attendance.student_id
            ORDER BY attendance.id DESC LIMIT 10"""
    ).fetchall()
    conn.close()
    return render_template(
        "dashboard.html",
        total_students=students,
        present_today=present_today,
        recent=recent,
    )


@app.route("/add_student", methods=["GET", "POST"])
@login_required
def add_student():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        bus_no = request.form.get("bus_no", "").strip()
        image_data = request.form.get("image_data", "")

        if not (name and roll_no and bus_no):
            flash("All fields are required.", "error")
            return redirect(url_for("add_student"))

        conn = get_db()
        existing = conn.execute(
            "SELECT id FROM students WHERE roll_no = ?", (roll_no,)
        ).fetchone()
        if existing:
            conn.close()
            flash("A student with this roll number already exists.", "error")
            return redirect(url_for("add_student"))

        image_path = None
        try:
            from face_recognition import face_encodings as fr_encodings
        except ImportError:
            fr_encodings = None

        if image_data and fr_encodings:
            try:
                img_bytes = base64.b64decode(image_data.split(",")[1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = fr_encodings(rgb)
                if not encs:
                    flash("No face detected. Please use a clear photo.", "error")
                    conn.close()
                    return redirect(url_for("add_student"))

                # Save student image to dataset
                student_dir = os.path.join(DATASET_DIR, roll_no)
                os.makedirs(student_dir, exist_ok=True)
                image_path = os.path.join(student_dir, f"{roll_no}.jpg")
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
            "INSERT INTO students (name, roll_no, bus_no, face_encoding, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, roll_no, bus_no, enc_bytes, image_path),
        )
        conn.commit()
        conn.close()
        flash(f"Student {name} added successfully.", "success")
        return redirect(url_for("add_student"))

    return render_template("add_student.html")


@app.route("/students")
@login_required
def students():
    conn = get_db()
    rows = conn.execute("SELECT * FROM students ORDER BY roll_no").fetchall()
    conn.close()
    return render_template("attendance.html", students=rows, mode="students")


@app.route("/delete_student/<int:student_id>", methods=["POST"])
@login_required
def delete_student(student_id):
    conn = get_db()
    student = conn.execute(
        "SELECT id, name, roll_no, image_path, face_encoding FROM students WHERE id = ?",
        (student_id,),
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


@app.route("/live_attendance")
@login_required
def live_attendance():
    return render_template("live_attendance.html")


@app.route("/attendance")
@login_required
def attendance():
    conn = get_db()
    rows = conn.execute(
        """SELECT attendance.date, attendance.time, attendance.bus_no,
                  attendance.status, students.name, students.roll_no
            FROM attendance
            JOIN students ON students.id = attendance.student_id
            ORDER BY attendance.id DESC"""
    ).fetchall()
    conn.close()
    return render_template("attendance.html", records=rows, mode="attendance")


# ---------------------------------------------------------------------------
# Live recognition (video + capture callback)
# ---------------------------------------------------------------------------
def gen_frames():
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
            encodings = load_encodings()
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
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/mark_attendance", methods=["POST"])
@login_required
def mark_attendance():
    global live_student_name, mark_done
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image_data") or request.form.get("image_data") or ""
    student_name = None

    if image_data:
        try:
            img_bytes = base64.b64decode(image_data.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                student_name = find_matching_student(img)
        except Exception:
            student_name = None
    elif live_student_name and not mark_done:
        student_name = live_student_name

    if student_name:
        conn = get_db()
        student = conn.execute(
            "SELECT * FROM students WHERE name = ?", (student_name,)
        ).fetchone()
        if student:
            today = date.today().isoformat()
            now = datetime.now().strftime("%H:%M:%S")
            existing = conn.execute(
                "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
                (student["id"], today),
            ).fetchone()
            if existing:
                conn.close()
                return jsonify({"status": "already", "name": student_name})
            conn.execute(
                "INSERT INTO attendance (student_id, bus_no, date, time) "
                "VALUES (?, ?, ?, ?)",
                (student["id"], student["bus_no"], today, now),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)

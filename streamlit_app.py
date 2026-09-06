import os
from datetime import date, datetime

import cv2
import numpy as np
import streamlit as st

import app as flask_app


st.set_page_config(page_title="Bus Attendance", page_icon="🚌", layout="wide")
flask_app.init_db()


def decode_image(uploaded_file):
    if uploaded_file is None:
        return None
    image = cv2.imdecode(
        np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    return image


def login_form():
    st.title("🚌 Bus Attendance")
    st.caption("Admin Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")
    if submitted:
        conn = flask_app.get_db()
        admin = conn.execute(
            "SELECT id FROM admin WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        conn.close()
        if admin:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Invalid username or password")


def attendance_session():
    minutes = datetime.now().hour * 60 + datetime.now().minute
    if 360 <= minutes < 660:
        return "Morning"
    if 900 <= minutes < 1140:
        return "Evening"
    return None


def dashboard():
    today = date.today().isoformat()
    conn = flask_app.get_db()
    total = conn.execute("SELECT COUNT(*) AS c FROM students").fetchone()["c"]
    present = conn.execute(
        "SELECT COUNT(DISTINCT student_id) AS c FROM attendance WHERE date = ?",
        (today,),
    ).fetchone()["c"]
    absent = conn.execute(
        """SELECT id, name, roll_no, bus_no FROM students
           WHERE NOT EXISTS (
             SELECT 1 FROM attendance
             WHERE attendance.student_id = students.id AND attendance.date = ?
           ) ORDER BY bus_no, name""",
        (today,),
    ).fetchall()
    conn.close()

    st.title("Dashboard")
    first, second, third = st.columns(3)
    first.metric("Total Students", total)
    second.metric("Present Today", present)
    third.metric("Absent Today", len(absent))

    st.subheader("Absent Students Today")
    if absent:
        st.dataframe(
            [dict(row) | {"status": "Absent"} for row in absent],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("All registered students are present today.")


def add_student():
    st.title("Add Student")
    st.info("Upload the front of the college ID card or capture a clear face photo. Enter the printed details below.")
    with st.form("add_student_form"):
        name = st.text_input("Name")
        roll_no = st.text_input("Roll No")
        bus_no = st.text_input("Bus No")
        id_card = st.file_uploader("College ID Card", type=["jpg", "jpeg", "png"])
        camera = st.camera_input("Capture Face Photo")
        submitted = st.form_submit_button("Add Student", type="primary")

    if not submitted:
        return
    if not (name.strip() and roll_no.strip() and bus_no.strip()):
        st.error("Name, roll number, and bus number are required.")
        return
    if id_card is None and camera is None:
        st.error("Upload an ID card or capture a face photo.")
        return

    image = decode_image(id_card or camera)
    if image is None:
        st.error("The selected image could not be read.")
        return
    try:
        import face_recognition

        encodings = face_recognition.face_encodings(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except ImportError:
        encodings = []
    if not encodings:
        st.error("No face was detected. Use a clear card or face photo.")
        return

    conn = flask_app.get_db()
    if conn.execute("SELECT id FROM students WHERE roll_no = ?", (roll_no.strip(),)).fetchone():
        conn.close()
        st.error("That roll number is already registered.")
        return

    student_dir = os.path.join(flask_app.DATASET_DIR, roll_no.strip())
    os.makedirs(student_dir, exist_ok=True)
    image_path = os.path.join(student_dir, f"{roll_no.strip()}_id_card.jpg")
    cv2.imwrite(image_path, image)
    encoding = encodings[0]
    all_encodings = flask_app.load_encodings()
    all_encodings[roll_no.strip()] = {
        "name": name.strip(), "roll_no": roll_no.strip(), "encoding": encoding
    }
    flask_app.save_encodings(all_encodings)
    conn.execute(
        "INSERT INTO students (name, roll_no, bus_no, face_encoding, image_path) VALUES (?, ?, ?, ?, ?)",
        (name.strip(), roll_no.strip(), bus_no.strip(), flask_app.pickle.dumps(encoding), image_path),
    )
    conn.commit()
    conn.close()
    st.success(f"Student {name.strip()} added successfully.")


def live_attendance():
    st.title("Live Attendance")
    session = attendance_session()
    if not session:
        st.warning("Attendance is open from 6:00 AM to 11:00 AM and 3:00 PM to 7:00 PM.")
        return
    st.caption(f"{session} attendance is open")
    camera = st.camera_input("Capture the student's face")
    if camera is None or not st.button("Mark Attendance", type="primary"):
        return

    image = decode_image(camera)
    student_name = flask_app.find_matching_student(image)
    if not student_name:
        st.warning("Unknown person. Use the Unknown Persons log to review the captured entry.")
        return

    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")
    conn = flask_app.get_db()
    student = conn.execute("SELECT * FROM students WHERE name = ?", (student_name,)).fetchone()
    existing = conn.execute(
        "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
        (student["id"], today),
    ).fetchone()
    if existing:
        conn.close()
        st.info(f"Already marked today: {student_name}")
        return
    image_path = os.path.relpath(student["image_path"], flask_app.BASE_DIR) if student["image_path"] else None
    conn.execute(
        "INSERT INTO attendance (student_id, bus_no, date, time, image_path) VALUES (?, ?, ?, ?, ?)",
        (student["id"], student["bus_no"], today, now, image_path),
    )
    conn.commit()
    conn.close()
    st.success(f"Marked present: {student_name}")


def logs(unknown=False):
    title = "Unknown Persons" if unknown else "Attendance Log"
    st.title(title)
    conn = flask_app.get_db()
    if unknown:
        rows = conn.execute("SELECT * FROM unknown_attendance ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute(
            """SELECT attendance.id, attendance.date, attendance.time,
                      attendance.bus_no, attendance.status, attendance.image_path,
                      students.name, students.roll_no
               FROM attendance JOIN students ON students.id = attendance.student_id
               ORDER BY attendance.id DESC"""
        ).fetchall()
    conn.close()
    if not rows:
        st.info("No records yet.")
        return
    for row in rows:
        values = dict(row)
        with st.container(border=True):
            columns = st.columns([2, 2, 2, 2, 2, 1])
            columns[0].write(f"**{values.get('name', 'Unknown Person')}**")
            columns[1].write(f"Roll: {values.get('roll_no', 'Unknown')}")
            columns[2].write(f"Bus: {values.get('bus_no', 'Unknown')}")
            columns[3].write(f"{values['date']} {values['time']}")
            columns[4].write(values.get("status", "unknown").title())
            endpoint = "delete_unknown_attendance" if unknown else "delete_attendance"
            if columns[5].button("Delete", key=f"delete_{endpoint}_{values['id']}"):
                conn = flask_app.get_db()
                table = "unknown_attendance" if unknown else "attendance"
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (values["id"],))
                conn.commit()
                conn.close()
                st.rerun()
            if values.get("image_path"):
                image_path = os.path.join(flask_app.BASE_DIR, values["image_path"])
                if os.path.exists(image_path):
                    st.image(image_path, width=160)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_form()
    st.stop()

with st.sidebar:
    st.title("Bus Attendance")
    page = st.radio("Navigate", ["Dashboard", "Add Student", "Live Attendance", "Attendance Log", "Unknown Persons"])
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

if page == "Dashboard":
    dashboard()
elif page == "Add Student":
    add_student()
elif page == "Live Attendance":
    live_attendance()
elif page == "Attendance Log":
    logs()
else:
    logs(unknown=True)

import os
import pickle
import sqlite3
from datetime import date, datetime

import cv2
import numpy as np
import streamlit as st

from app import (
    CAPTURES_DIR,
    DATASET_DIR,
    STORAGE_DIR,
    get_db,
    init_db,
    load_encodings,
    save_encodings,
)


st.set_page_config(page_title="Bus Attendance", page_icon="🚌", layout="wide")
init_db()


def query_all(sql, params=()):
    conn = get_db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def query_one(sql, params=()):
    conn = get_db()
    row = conn.execute(sql, params).fetchone()
    conn.close()
    return row


def scoped_clause(bus_no, column="bus_no"):
    return (f" WHERE {column} = ?", (bus_no,)) if bus_no else ("", ())


def login_admin(username, password):
    return query_one(
        "SELECT * FROM admin WHERE username = ? AND password = ?",
        (username.strip(), password),
    )


def login_student(roll_no, password):
    return query_one(
        "SELECT id, name, roll_no FROM students WHERE roll_no = ? AND password = ?",
        (roll_no.strip(), password),
    )


def logout():
    for key in ("role", "username", "bus_no", "student_id", "student_name", "student_roll_no"):
        st.session_state.pop(key, None)


def save_student_face(uploaded_file, roll_no, name):
    if uploaded_file is None:
        return None, None
    try:
        import face_recognition

        image_bytes = uploaded_file.getvalue()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The image could not be read.")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            raise ValueError("No face was detected in the uploaded image.")

        student_dir = os.path.join(DATASET_DIR, roll_no)
        os.makedirs(student_dir, exist_ok=True)
        image_path = os.path.join(student_dir, f"{roll_no}_id_card.jpg")
        cv2.imwrite(image_path, image)

        known = load_encodings()
        known[roll_no] = {"name": name, "roll_no": roll_no, "encoding": encodings[0]}
        save_encodings(known)
        return image_path, pickle.dumps(encodings[0])
    except ImportError as exc:
        raise ValueError("Face recognition is not installed on this deployment.") from exc


def add_student(form, uploaded_file):
    name = form["name"].strip()
    roll_no = form["roll_no"].strip()
    bus_no = st.session_state.get("bus_no") or form["bus_no"].strip()
    required = (name, roll_no, form["date_of_birth"], form["boarding_point"], form["year_of_study"], bus_no)
    if not all(required):
        st.error("Complete all required fields.")
        return
    try:
        fees_amount = float(form["fees_amount"])
        if fees_amount < 0:
            raise ValueError
    except ValueError:
        st.error("Fees amount must be a valid non-negative number.")
        return

    if query_one("SELECT id FROM students WHERE roll_no = ?", (roll_no,)):
        st.error("That register number already exists.")
        return

    image_path = None
    encoding_bytes = None
    if uploaded_file is not None:
        try:
            image_path, encoding_bytes = save_student_face(uploaded_file, roll_no, name)
        except ValueError as exc:
            st.error(str(exc))
            return

    conn = get_db()
    conn.execute(
        "INSERT INTO students (name, roll_no, date_of_birth, password, boarding_point, fees_amount, year_of_study, bus_no, face_encoding, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, roll_no, form["date_of_birth"], form["password"] or roll_no, form["boarding_point"], fees_amount, form["year_of_study"], bus_no, encoding_bytes, image_path),
    )
    conn.commit()
    conn.close()
    st.success(f"Student {name} added successfully.")


def mark_attendance(student_id, status):
    student = query_one("SELECT id, name, bus_no FROM students WHERE id = ?", (student_id,))
    if not student:
        st.error("Student was not found.")
        return
    today = date.today().isoformat()
    if query_one("SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id, today)):
        st.warning(f"Attendance is already marked for {student['name']} today.")
        return
    conn = get_db()
    conn.execute(
        "INSERT INTO attendance (student_id, bus_no, date, time, status) VALUES (?, ?, ?, ?, ?)",
        (student["id"], student["bus_no"], today, datetime.now().strftime("%H:%M:%S"), status),
    )
    conn.commit()
    conn.close()
    st.success(f"{student['name']} marked {status}.")


def render_login():
    st.title("🚌 Bus Attendance")
    st.caption("Secure attendance management for bus administrators and students.")
    admin_tab, student_tab = st.tabs(["Admin login", "Student login"])
    with admin_tab:
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            admin = login_admin(username, password)
            if admin:
                st.session_state.update(role="admin", username=admin["username"], bus_no=admin["bus_no"] or None)
                st.rerun()
            st.error("Invalid username or password.")
    with student_tab:
        with st.form("student_login"):
            roll_no = st.text_input("Register number")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            student = login_student(roll_no, password)
            if student:
                st.session_state.update(role="student", student_id=student["id"], student_name=student["name"], student_roll_no=student["roll_no"])
                st.rerun()
            st.error("Invalid register number or password.")


def render_dashboard():
    bus_no = st.session_state.get("bus_no")
    clause, params = scoped_clause(bus_no)
    today = date.today().isoformat()
    total = query_one(f"SELECT COUNT(*) AS count FROM students{clause}", params)["count"]
    present_params = (today, bus_no) if bus_no else (today,)
    present = query_one("SELECT COUNT(DISTINCT student_id) AS count FROM attendance WHERE date = ? AND status = 'present'" + (" AND bus_no = ?" if bus_no else ""), present_params)["count"]
    absent = query_all("SELECT name, roll_no, bus_no FROM students WHERE id NOT IN (SELECT student_id FROM attendance WHERE date = ? AND status = 'present')" + (" AND bus_no = ?" if bus_no else "") + " ORDER BY bus_no, name", present_params)
    st.title("Dashboard")
    first, second, third = st.columns(3)
    first.metric("Total students", total)
    second.metric("Present today", present)
    third.metric("Absent today", len(absent))
    st.subheader("Absent students today")
    st.dataframe([dict(row) for row in absent], use_container_width=True, hide_index=True)
    recent = query_all("SELECT attendance.date, attendance.time, students.name, students.roll_no, attendance.bus_no, attendance.status FROM attendance JOIN students ON students.id = attendance.student_id" + (" WHERE attendance.bus_no = ?" if bus_no else "") + " ORDER BY attendance.id DESC LIMIT 10", (bus_no,) if bus_no else ())
    st.subheader("Recent attendance")
    st.dataframe([dict(row) for row in recent], use_container_width=True, hide_index=True)


def render_add_student():
    st.title("Add student")
    with st.form("add_student"):
        left, right = st.columns(2)
        name = left.text_input("Full name *")
        roll_no = left.text_input("Register number *")
        date_of_birth = left.date_input("Date of birth *", value=None)
        boarding_point = left.text_input("Boarding point *")
        year_of_study = right.text_input("Year of study *")
        fees_amount = right.text_input("Fees amount *")
        password = right.text_input("Student password", help="Defaults to the register number when blank.")
        bus_no = right.text_input("Bus number *", disabled=bool(st.session_state.get("bus_no")), value=st.session_state.get("bus_no") or "")
        photo = st.file_uploader("Face or ID image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Add student", type="primary")
    if submitted:
        form = {"name": name, "roll_no": roll_no, "date_of_birth": date_of_birth.isoformat() if date_of_birth else "", "boarding_point": boarding_point, "year_of_study": year_of_study, "fees_amount": fees_amount, "password": password, "bus_no": bus_no}
        add_student(form, photo)


def render_students():
    st.title("Students")
    bus_no = st.session_state.get("bus_no")
    clause, params = scoped_clause(bus_no)
    rows = query_all("SELECT id, name, roll_no, date_of_birth, boarding_point, fees_amount, year_of_study, bus_no FROM students" + clause + " ORDER BY bus_no, name", params)
    for row in rows:
        with st.expander(f"{row['name']} · {row['roll_no']} · Bus {row['bus_no']}"):
            st.write({key: row[key] for key in row.keys() if key != "id"})
            if st.button("Delete student", key=f"delete_{row['id']}"):
                conn = get_db()
                conn.execute("DELETE FROM attendance WHERE student_id = ?", (row["id"],))
                conn.execute("DELETE FROM students WHERE id = ?", (row["id"],))
                conn.commit()
                conn.close()
                encodings = load_encodings()
                encodings.pop(row["roll_no"], None)
                save_encodings(encodings)
                st.rerun()


def render_attendance():
    st.title("Attendance log")
    bus_no = st.session_state.get("bus_no")
    sql = "SELECT attendance.id, attendance.date, attendance.time, attendance.bus_no, attendance.status, students.name, students.roll_no, students.year_of_study, students.boarding_point FROM attendance JOIN students ON students.id = attendance.student_id"
    rows = query_all(sql + (" WHERE attendance.bus_no = ?" if bus_no else "") + " ORDER BY attendance.id DESC", (bus_no,) if bus_no else ())
    st.dataframe([dict(row) for row in rows], use_container_width=True, hide_index=True)


def render_live_attendance():
    st.title("Live attendance")
    if st.session_state.get("role") == "student":
        st.info(f"Signed in as {st.session_state['student_name']}. Verify your details, then choose your status.")
    with st.form("verify_attendance"):
        roll_no = st.text_input("Register number", value=st.session_state.get("student_roll_no", ""), disabled=st.session_state.get("role") == "student")
        dob = st.text_input("Date of birth", placeholder="YYYY-MM-DD")
        submitted = st.form_submit_button("Verify student", type="primary")
    if submitted:
        bus_no = st.session_state.get("bus_no")
        student = query_one("SELECT id, name, roll_no, date_of_birth, bus_no FROM students WHERE roll_no = ? AND date_of_birth = ?" + (" AND bus_no = ?" if bus_no else ""), (roll_no, dob, bus_no) if bus_no else (roll_no, dob))
        if student and (st.session_state.get("role") != "student" or student["id"] == st.session_state.get("student_id")):
            st.session_state["verified_student"] = dict(student)
        else:
            st.error("The register number and date of birth do not match.")
    student = st.session_state.get("verified_student")
    if student:
        st.success(f"Verified: {student['name']} ({student['roll_no']})")
        present, absent = st.columns(2)
        if present.button("Mark present", type="primary"):
            mark_attendance(student["id"], "present")
        if absent.button("Mark absent"):
            mark_attendance(student["id"], "absent")


def render_admin_management():
    st.title("Bus administrators")
    with st.form("add_admin"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        bus_no = st.text_input("Bus number")
        submitted = st.form_submit_button("Create admin", type="primary")
    if submitted:
        if not all((username.strip(), password, bus_no.strip())):
            st.error("All fields are required.")
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO admin (username, password, bus_no) VALUES (?, ?, ?)", (username.strip(), password, bus_no.strip()))
                conn.commit()
                st.success("Bus administrator created.")
            except sqlite3.IntegrityError:
                st.error("That username already exists.")
            finally:
                conn.close()
    rows = query_all("SELECT username, bus_no FROM admin WHERE bus_no != '' ORDER BY bus_no, username")
    st.dataframe([dict(row) for row in rows], use_container_width=True, hide_index=True)


def render_app():
    role = st.session_state.get("role")
    if not role:
        render_login()
        return
    with st.sidebar:
        st.title("🚌 Bus Attendance")
        st.caption(st.session_state.get("username") or st.session_state.get("student_name"))
        if role == "admin":
            page = st.radio("Navigation", ["Dashboard", "Add student", "Students", "Live attendance", "Attendance log"] + (["Bus administrators"] if not st.session_state.get("bus_no") else []))
        else:
            page = "Live attendance"
        if st.button("Log out"):
            logout()
            st.rerun()
    {"Dashboard": render_dashboard, "Add student": render_add_student, "Students": render_students, "Live attendance": render_live_attendance, "Attendance log": render_attendance, "Bus administrators": render_admin_management}[page]()


render_app()

# Bus Attendance

A Flask-based bus attendance system that uses face recognition to register students, mark attendance automatically, and log unknown people captured by the live camera.

## Repository

[View the Bus Attendance repository on GitHub](https://github.com/praveenkumar21122006/bus-attendance)

## Features

- Admin login
- Add and delete students
- Store student names, roll numbers, bus numbers, and face photos
- Automatic face recognition from the browser camera
- Attendance records with captured photos
- Unknown-person detection and photo logging
- Students grouped and ordered by bus number
- Attendance and unknown-person logs

## Local Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/praveenkumar21122006/bus-attendance.git
cd bus-attendance
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open <http://localhost:5000> in a browser.

Default login:

- Username: `admin`
- Password: `admin123`

Change the default credentials before production use.

## Production Start

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app
```

The included `railway.toml` configures deployment on Railway. Set a secure `SECRET_KEY` environment variable and use persistent storage for `database.db`, `dataset/`, `captures/`, and `face_encodings.pkl`.

## Browser Camera Access

The live camera requires browser permission and normally works on `localhost` or an HTTPS deployment.

## Streamlit Cloud

The Streamlit version is in `streamlit_app.py`.

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select this GitHub repository and the `main` branch.
3. Set the main file to `streamlit_app.py`.
4. Deploy the app.

Use the default login `admin` / `admin123` after deployment. Streamlit Cloud storage is not permanent, so use an external database and object storage for production attendance data and photos.

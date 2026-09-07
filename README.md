# Bus Attendance

A Flask-based bus attendance system that uses face recognition to register students, mark attendance automatically, and log unknown people captured by the live camera.

## Repository

[View the Bus Attendance repository on GitHub](https://github.com/praveenkumar21122006/bus-attendance)

## Features

- Separate admin and student logins
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

Student login:

- Open `/student-login` or use the link on the admin login page.
- Students use their roll number and the password assigned when the admin adds them.
- If the admin leaves the password blank, the student's initial password is their roll number.

Change the default credentials before production use.

## Production Start

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app
```

Set a secure `SECRET_KEY` environment variable and use persistent storage for `database.db`, `dataset/`, `captures/`, and `face_encodings.pkl`.

## Permanent Render Deployment

The included `render.yaml` deploys the Docker image as a Render web service and mounts a persistent disk at `/var/lib/bus-attendance`. This keeps the SQLite database, student photos, attendance captures, and face encodings across deploys. Persistent disks require Render's Starter plan or higher.

1. Push this repository to GitHub.
2. In [Render](https://render.com/), choose **New > Blueprint** and select the repository.
3. Confirm the `render.yaml` configuration and create the service.
4. Open the generated `onrender.com` URL.

Render generates `SECRET_KEY` automatically. Change the default admin password immediately after the first login.

## Railway Deployment

Railway can deploy the same Docker image and provides a persistent volume for attendance data.

1. Open [Railway](https://railway.app/) and choose **Deploy from GitHub repo**.
2. Select `praveenkumar21122006/bus-attendance`.
3. Add a Railway volume mounted at `/data`.
4. Add a `SECRET_KEY` variable with a long random value.
5. Deploy the service and open the generated public domain.

The included `railway.toml` configures Docker, Gunicorn, health checks, and `DATA_DIR=/data`.

## Hugging Face Spaces Deployment

For a free public demo, create a new Hugging Face Space with the **Docker** SDK and select a public repository. Use these settings:

- Repository: `praveenkumar21122006/bus-attendance`
- Branch: `main`
- Docker port: `7860`

The Space will build the included `Dockerfile` and provide an HTTPS URL. Add a `SECRET_KEY` Space variable before using the app. Free Space storage is temporary, so use an external database and object storage for permanent attendance records and photos.

## PythonAnywhere Deployment

PythonAnywhere can host the Flask app with persistent files on its free web-app plan.

1. Create a PythonAnywhere account and open a **Bash console**.
2. Clone the repository:

```bash
git clone https://github.com/praveenkumar21122006/bus-attendance.git
cd bus-attendance
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Open the **Web** tab, create a manual Flask web app, and choose Python 3.10.
4. Set the WSGI file to `/home/YOUR_USERNAME/bus-attendance/wsgi.py`.
5. In the WSGI file, set `SECRET_KEY` in the environment before importing `app`.
6. Reload the web app and open the generated `YOUR_USERNAME.pythonanywhere.com` URL.

The `wsgi.py` entrypoint stores the database, face encodings, datasets, and captures under the project `data/` directory. Free accounts have CPU, storage, and package-installation limits; face recognition may be slow.

## Browser Camera Access

The live camera requires browser permission and normally works on `localhost` or an HTTPS deployment.


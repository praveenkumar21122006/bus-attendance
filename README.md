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

## Firebase Deployment

Firebase Hosting serves the HTTPS URL and forwards application requests to a Cloud Run container. Install and authenticate the Firebase and Google Cloud CLIs, then run these commands from the project directory:

```bash
firebase login
gcloud auth login
gcloud config set project YOUR_FIREBASE_PROJECT_ID
gcloud builds submit --tag gcr.io/YOUR_FIREBASE_PROJECT_ID/bus-attendance
gcloud run deploy bus-attendance \
	--image gcr.io/YOUR_FIREBASE_PROJECT_ID/bus-attendance \
	--region us-central1 \
	--platform managed \
	--allow-unauthenticated \
	--set-env-vars SECRET_KEY=REPLACE_WITH_A_LONG_RANDOM_VALUE
firebase deploy --only hosting
```

The `firebase.json` rewrite expects the Cloud Run service to be named `bus-attendance` in `us-central1`. Cloud Run's local filesystem is temporary, so this deployment is suitable for a demo but not durable attendance data. Move SQLite, face encodings, and uploaded images to a persistent database and Cloud Storage before production use.

## Browser Camera Access

The live camera requires browser permission and normally works on `localhost` or an HTTPS deployment.

## Streamlit Cloud

The Streamlit version is in `streamlit_app.py`.

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select this GitHub repository and the `main` branch.
3. Set the main file to `streamlit_app.py`.
4. Deploy the app.

Use the default login `admin` / `admin123` after deployment. Streamlit Cloud storage is not permanent, so use an external database and object storage for production attendance data and photos.

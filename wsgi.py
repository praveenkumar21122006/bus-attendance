import os

os.environ.setdefault("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

from app import app as application

import os
import sys

# Tambahkan path aplikasi ke sys.path
sys.path.insert(0, os.path.dirname(__file__))

from a2wsgi import ASGIMiddleware
from main import app

# Konversi ASGI (FastAPI) ke WSGI agar didukung oleh Phusion Passenger di cPanel
application = ASGIMiddleware(app)

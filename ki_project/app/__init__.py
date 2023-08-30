# __init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config  # Importiere die Konfiguration
import os

template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')


db = SQLAlchemy()

def create_app():
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config.from_object(Config)  # Lade die Konfiguration

    db.init_app(app)

    with app.app_context():  # Stelle sicher, dass die Datenbanktabellen erstellt sind
        db.create_all()

    from .routes import main
    app.register_blueprint(main)

    return app

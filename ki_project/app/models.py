# models.py

from datetime import datetime
from . import db  # Importiere db aus unserem Projekt

class RatedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(120), nullable=False)
    rating = db.Column(db.String(20), nullable=False)
    date_rated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"RatedImage('{self.image_path}', '{self.rating}')"

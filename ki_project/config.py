# config.py

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///rated_image.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'eine_very_geheime_key'

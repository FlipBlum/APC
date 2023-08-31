# forms.py

from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from flask_wtf.file import FileField, FileRequired
from constanten import CLASS_LABELS

class ClassificationForm(FlaskForm):
    classification = RadioField('Bewertung', choices=CLASS_LABELS)
    submit = SubmitField('Absenden')
    image = FileField('Bilder hochladen', validators=[FileRequired()], render_kw={"multiple": True})

# forms.py

from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from flask_wtf.file import FileField, FileRequired

CLASS_LABELS = [('gut', 'Gut'), ('schlecht', 'Schlecht')]
CLASSES = [label[0] for label in CLASS_LABELS]
class ClassificationForm(FlaskForm):
    classification = RadioField('Bewertung', choices=CLASS_LABELS)
    submit = SubmitField('Absenden')
    image = FileField('Bilder hochladen', validators=[FileRequired()], render_kw={"multiple": True})

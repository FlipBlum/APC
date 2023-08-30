# forms.py

from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from flask_wtf.file import FileField, FileRequired

class ClassificationForm(FlaskForm):
    classification = RadioField('Bewertung', choices=[('daisy', 'Daisy'), ('dandelion', 'Löwenzahn'), ('roses', 'Rose'), ('sunflower', 'Sonnenblume'), ('tulips', 'Tulpe')])
    submit = SubmitField('Absenden')
    image = FileField('Bilder hochladen', validators=[FileRequired()], render_kw={"multiple": True})

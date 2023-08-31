from flask import Blueprint, render_template, redirect, url_for, request
from .models import RatedImage
from . import db
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from collections import namedtuple
from kiModel import SimpleCNN
import torch


# Google Drive Authentifizierung
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

main = Blueprint('main', __name__)
train_path = "/Users/philippblum/Desktop/coding/ki_project/static/images/train"
validate_path = "/Users/philippblum/Desktop/coding/ki_project/static/images/validate"
folder_id = '1ctX_8ylkA23XeHJgJFbxVMieCZedf3Cg'


def fetch_images_from_drive(folder_id, image_folder):
    query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    for file in file_list:
        # Hier speichern Sie das Bild temporär in Ihrem Flask static/images Ordner
        file.GetContentFile(os.path.join(image_folder, file['title']))

@main.route('/', methods=['GET', 'POST'])
def index():
    image_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'images', 'classify')

    # Holt alle Bilder aus Google Drive
    fetch_images_from_drive('1ctX_8ylkA23XeHJgJFbxVMieCZedf3Cg', image_folder)

    Image = namedtuple('Image', ['path', 'alt'])
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    images = [Image(url_for('static', filename=f'images/classify/{f}'), f'Image {f}') for f in image_files]


    if request.method == 'POST':
        classifications = request.form.getlist("classification")  # Alle Klassifizierungen holen

        for idx, image_file in enumerate(image_files):
            rating = classifications[idx]

            if (idx + 1) % 5 == 0:
                base_path = validate_path
            else:
                base_path = train_path
                
            # Pfad zum richtigen Ordner basierend auf der Klassifizierung erstellen
            target_folder = os.path.join(base_path, rating)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # Bild in den entsprechenden Ordner verschieben
            src = os.path.join(image_folder, image_file)
            dst = os.path.join(target_folder, image_file)
            os.rename(src, dst)

            # Pfad zur Datenbank hinzufügen
            relative_path = os.path.join('flower_photos', rating, image_file)  # Für die DB
            rated_image = RatedImage(image_path=relative_path, rating=rating)
            db.session.add(rated_image)
            db.session.commit()

            print(f"Bild {rating} klassifiziert")

        return redirect(url_for('main.index'))

    return render_template('index.html', images=images)

@app.route('/predict', methods=['POST'])
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()  # Erzeuge eine Instanz des Modells
    model.load_state_dict(torch.load("flowerprediction_v2.pth"))  # Lade das state dictionary
    model.to(device)
    model.eval()

    image_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'images', 'predict')
    Image = namedtuple('Image', ['path', 'alt', 'prediction'])
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    images_with_predictions = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # Hier würden Sie Ihren Vorverarbeitungscode und den Inferenzcode hinzufügen, 
        # um Vorhersagen für das Bild zu erhalten. Zum Beispiel:
        # image_tensor = preprocess_image(image_path) 
        # prediction = model(image_tensor)
        
        # Zum Zweck dieses Beispiels werde ich einfach einen Dummy-Wert hinzufügen:
        prediction = "Flower Type X"
        
        image = Image(url_for('static', filename=f'images/predict/{image_file}'), f'Image {image_file}', prediction)
        images_with_predictions.append(image)

    return render_template('predict.html', images=images_with_predictions)
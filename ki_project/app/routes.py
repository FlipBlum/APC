from flask import Blueprint, render_template, redirect, url_for, request
from .models import RatedImage
from . import db
import os
from collections import namedtuple
from simpleCNN import SimpleCNN
import torch
from PIL import Image
from torchvision import transforms
from constanten import CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()  # Erzeuge eine Instanz des Modells
model.load_state_dict(torch.load("ringprediction_v2.pth"))  # Lade das state dictionary
model.to(device)
model.eval()

main = Blueprint('main', __name__)
train_path = "/Users/philippblum/Desktop/coding/ki_project/static/images/train"
validate_path = "/Users/philippblum/Desktop/coding/ki_project/static/images/validate"

def get_prediction(model, image_tensor, device):
    image_tensor = image_tensor.to(device)  # send the tensor to the device (GPU or CPU)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = outputs.max(1)
    return predicted_class.item()

def preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((180, 180)), transforms.ToTensor()])
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # da wir nur ein Bild haben, fügen wir eine Dimension hinzu
    return image_tensor

@main.route('/', methods=['GET', 'POST'])
def index():
    image_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'images', 'classify')

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

@main.route('/predict', methods=['POST'])
def predict():
    image_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'images', 'predict')
    Image = namedtuple('Image', ['path', 'alt', 'prediction'])
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    images_with_predictions = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        try:
            image_tensor = preprocess_image(image_path)
            prediction_index = get_prediction(model, image_tensor, device)
            prediction_label = CLASSES[prediction_index]  # Der Name der Klasse basierend auf dem Index
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {image_file}: {e}")
            prediction_label = "Fehler"
        
        image = Image(url_for('static', filename=f'images/predict/{image_file}'), f'Image {image_file}', prediction_label)
        images_with_predictions.append(image)

    return render_template('predict.html', images=images_with_predictions)
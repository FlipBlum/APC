import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps
from kiModel import SimpleCNN

st.title("A Flower Image Recognition App")

st.title("Making Predictions")
st.write("""Please upload your choice of flower to predict""")
uploaded_file = st.file_uploader("Choose a jpeg file", type=["jfif", "jpg", "jpeg"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN() # Erzeuge eine Instanz des Modells
model.load_state_dict(torch.load("flowerprediction_v2.pth")) # Lade das state dictionary
model.to(device)
model.eval()


if uploaded_file is None:
    st.write("Please upload a jpg, jpeg or a jfif image of the flower in the drag and drop box above")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    size = (180, 180)
    method = Image.NEAREST if image.size == size else Image.ANTIALIAS
    image = ImageOps.fit(image, size, method=method, centering=(0.5, 0.5))
    image = np.asarray(image)
    image = torch.Tensor(image.transpose((2, 0, 1))).unsqueeze(0) / 255.0  # PyTorch erwartet (C, H, W)
    st.write("Now click the button below to make prediction")

    if st.button('Make Prediction'):
        class_names = np.array(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
        with torch.no_grad():
            predictions = model(image.to(device))
            probability = F.softmax(predictions, dim=1)
            prediction = torch.argmax(probability, axis=1)
            confidence = round(torch.max(probability).item() * 100, 2)
        
        if prediction == 0 and confidence > 70:
            st.write(f"This is {class_names[0].upper()} and I am  {confidence}% about it")
        elif prediction == 1 and confidence > 70:
            st.write(f"This is {class_names[1].upper()} and I am  {confidence}% about it")
        elif prediction == 2 and confidence > 70:
            st.write(f"This is {class_names[2].upper()} and I am  {confidence}% about it")
        elif prediction == 3 and confidence > 70:
            st.write(f"This is {class_names[3].upper()} and I am  {confidence}% about it")
        elif prediction == 4 and confidence > 70:
            st.write(f"This is {class_names[4].upper()} and I am  {confidence}% about it")
        else:
            st.write("Sorry I have no idea what type of flower this is. Kindly upload another image of the same flower")

st.header("Developers Note")
st.write("Thank you so much for taking out time to interact with this APP. This App is intended to solve the little hassle with identifying some specific kind of flowers.")
st.header('Thank you so very Much!')
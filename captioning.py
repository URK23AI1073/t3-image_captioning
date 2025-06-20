import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained InceptionV3 model
def load_cnn_model():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

# Preprocess the image for the model
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Dummy caption (you can replace this with real model output later)
def generate_caption(features):
    return "A sunflower in full bloom under a golden sky."

# Main function
def caption_image(image_path):
    model = load_cnn_model()
    img_array = preprocess_image(image_path)
    features = model.predict(img_array, verbose=0)
    caption = generate_caption(features)

    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.axis("off")
    plt.show()

# Run the function
caption_image("sample.jpg")  # ← Replace with your image filename

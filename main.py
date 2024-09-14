import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def getPrediction(filename):
    # Class names for pneumonia classification (assuming it's a binary classification: pneumonia or no pneumonia)
    classes = ['No Pneumonia', 'Pneumonia']
    
    # Load the model for pneumonia classification
    my_model = load_model(r"C:\Users\biomedical research\Downloads\pneumonia_ml_web\model\model_vgg19.h5")

    
    # Resize image to the size that your model expects (assuming 224x224, adjust if needed)
    SIZE = 224
    img_path = 'static/images/' + filename
    
    # Open the image and convert it to RGB to ensure it has 3 channels (RGB)
    img = Image.open(img_path).convert('RGB')  # Convert to RGB
    img = img.resize((SIZE, SIZE))  # Resize the image
    
    # Convert image to array and normalize pixel values
    img = np.asarray(img) / 255.0
    
    # Ensure the image shape is (224, 224, 3)
    print("Image shape after conversion and resizing:", img.shape)
    
    # Expand dimensions to match input format for the model (batch size, height, width, channels)
    img = np.expand_dims(img, axis=0)
    
    # Ensure the input shape is (1, 224, 224, 3)
    print("Image shape before feeding to model:", img.shape)
    
    # Make a prediction
    pred = my_model.predict(img)
    
    # Convert prediction to class name
    pred_class = classes[np.argmax(pred)]
    print("Diagnosis is:", pred_class)
    return pred_class

from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from main import getPrediction
import os

# Save images to the 'static' folder as Flask serves images from this directory
UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create an app object using the Flask class
app = Flask(__name__, static_folder="static")

# Set secret key for session encryption
app.secret_key = "secret key"

# Define the upload folder to save images uploaded by the user
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def getPrediction(filename):
    # Class names for pneumonia classification (assuming it's a binary classification: pneumonia or no pneumonia)
    classes = ['No Pneumonia', 'Pneumonia']
    
    # Load the model for pneumonia classification
    my_model = load_model(r"model\model_vgg19.h5")

    
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

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

# Add Post method to the decorator to allow for form submission
@app.route('/', methods=['POST'])
def submit_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get prediction from model
            label = getPrediction(filename)
            full_filename = url_for('static', filename='images/' + filename)  # Correct URL for the image
            
            flash(f'Diagnosis: {label}')
            flash(full_filename)  # Pass the correct path for the image
        except Exception as e:
            flash(f'Error processing the image: {str(e)}')
            return redirect(request.url)
        
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Define port so we can map container port to localhost
    app.run(host='0.0.0.0', port=port)  # Define 0.0.0.0 for Docker

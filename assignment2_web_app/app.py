from flask import Flask, render_template, request, Response
import tensorflow as tf 
from keras.models import load_model
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
model = None

def load_nn_model():
    global model
    model_path = 'models/my_model.h5'
    model = load_model(model_path)
    model.summary()  # Optional: Display model summary

input_shape = (224, 224, 3)
num_classes = 6

def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((input_shape[1], input_shape[0]))

    # Convert the image to an array of floats between 0 and 1
    image = np.array(image) / 255.0

    # If the image has 3 channels, keep it as is. Otherwise, convert to 3 channels by duplicating the values
    if image.shape[-1] != input_shape[-1]:
        image = np.repeat(image[:, :, np.newaxis], input_shape[-1], axis=-1)

    # Return the preprocessed image
    return image

# Process the prediction result
def process_prediction(prediction):
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Define a list of class labels
    class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    # Get the predicted class label based on the index
    predicted_class_label = class_labels[predicted_class_index]

    # Create a response message with the predicted class label
    response = f"Predicted class: {predicted_class_label}"

    # Return the response
    return response

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if model is None:
            load_nn_model()  # Load the model if it's not already loaded
            if model is None:
                return "Model not found"

        # Check if the POST request contains a file
        if 'image' not in request.files:
            return "No image file found"

        # Read the uploaded image file
        file = request.files['image']
        if file.filename == '':
            return "No selected image file"

        try:
            # Load the image and preprocess it as required
            image = Image.open(file)
            image = preprocess_image(image)

            # Perform inference using the loaded model
            prediction = model.predict(np.expand_dims(image, axis=0))

            # Process the prediction and generate a response
            response = process_prediction(prediction)

            return Response(response, mimetype='text/plain')

        except Exception as e:
            return "Error processing image: " + str(e)

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()

from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import os

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load Keras model
def load_keras_model():
    global model
    try:
        model = tf.keras.models.load_model('plant2.hdf5')
        print(" * Model loaded successfully.")
    except Exception as e:
        print(" * Failed to load .h5 file. Error:", e)
        model = None

load_keras_model()

# Preprocess image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            imagefile = request.files['imagefile']
            image_path = './images/' + imagefile.filename
            imagefile.save(image_path)
            image = Image.open(imagefile)
            processed_image = preprocess_image(image, target_size=(256, 256))
            if model is None:
                load_keras_model()
                if model is None:
                    return render_template('index.html', prediction_result='Model could not be loaded')

            # Predict the image class
            prediction = model.predict(processed_image)
            class_name = ['Pepper_bell__Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
                          'Potato___Late_blight', 'Tomato_Bacterial_spot',
                          'Tomato_Early_blight', 'Tomato_Healthy',
                          'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_MosiacVirus',
                          'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_TargetSpot',
                          'Tomato_YellowLeaf',]
            predictions_array = np.array(prediction)

            # Find the index of the class with the highest probability
            predicted_class_index = np.argmax(predictions_array)

            # Retrieve the corresponding class label
            predicted_class_label = class_name[predicted_class_index]

            # Get the probability associated with the predicted class
            predicted_probability = predictions_array[0][predicted_class_index]

            # Output the result
            print("Predicted Class:", predicted_class_label)
            print("Predicted Probability:", predicted_probability)
            predicted_probability_percentage = round(predicted_probability * 100, 2)

            response = {
                'success': True,
                'prediction': predicted_class_label,
                'predicted_probability': predicted_probability_percentage  
            }
            return render_template('index.html', prediction_result=response)
        except Exception as e:
            return render_template('index.html', prediction_result=str(e))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)

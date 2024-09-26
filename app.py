from flask import Flask, render_template, request,send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

model_path = "C:\\Users\\naren\\OneDrive\\Desktop\\all\\Wheat disease\\epoch35.h5"
label_path = "C:\\Users\\naren\\OneDrive\\Desktop\\all\\Wheat disease\\labelmain"

model = load_model(model_path)
lb = pickle.load(open(label_path, "rb"))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    imagefile = request.files['imagefile']
    # Read the image using OpenCV
    image = cv2.imdecode(np.fromstring(imagefile.read(), np.uint8), cv2.IMREAD_COLOR)
    # Perform any required preprocessing on the image
    # Make predictions using the loaded model
    frame = cv2.resize(image, (224, 224)).astype("float32")
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    if 'Q' not in globals():
        Q = []

        # Append the prediction result to the Q list

    Q.append(preds)
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    prediction = label.upper()

    accuracy = results[i] * 100
    # Return the predictions
    return render_template('index.html', prediction=prediction, accuracy=accuracy)


@app.route('/Leafrust', methods=['GET'])
def leafrust_page():
    return render_template('Leafrust.html')

@app.route('/crown', methods=['GET'])
def crown_page():
    return render_template('crown.html')

@app.route('/losesmut', methods=['GET'])
def losesmut_page():
    return render_template('losesmut.html')



if __name__ == '__main__':
    app.run(port=3000, debug=True)

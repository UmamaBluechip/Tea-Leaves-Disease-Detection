import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  

model = load_model('model_1.h5')  

disease_names = ['Algal Leaf', 'Anthracnose', 'Bird Eye Spot', 'Brown Blight', 'Gray light', 'healthy', 'Red Leaf Spot', 'White Spot']  

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(150, 150)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction) 
    predicted_disease = disease_names[predicted_class_index]

    return predicted_disease

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                app.config['UPLOAD_FOLDER'],
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict_disease(image_location) 
            return render_template('result.html', prediction=pred, image_loc=image_file.filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 

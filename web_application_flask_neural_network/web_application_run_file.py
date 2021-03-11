import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'A_Random_Secret_key'

IMG_SIZE_NN = 150

def valid_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        # check for file in post request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # check for empty file, no filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # check if the file is valid for a required classification
        if file and valid_file(file.filename):
            filename = secure_filename(file.filename)

            # read file >> convert numpy array >> convert cv2 object
            # Reference: https://stackoverflow.com/questions/47515243/reading-image-file-file-storage-object-using-cv2
            filestr = request.files['file'].read()
            np_img = np.fromstring(filestr, np.uint8)
            # img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
            # cv2.imshow('a_picture', img)
            # cv2.waitKey(0)

            # load the saved/trained model
            # I did it with tensorflow, a simple neural network image classifier model
            # model saved as >>>model.save('models_nn_saved/model_4')
            model = tf.keras.models.load_model('models_nn_saved/model_4') # function argument: path to model_4 folder with .pb file

            # image processing that fits my model
            new_img = cv2.resize(img, (IMG_SIZE_NN, IMG_SIZE_NN))
            new_img = new_img.reshape(-1, IMG_SIZE_NN, IMG_SIZE_NN, 1)

            # model prediction
            prediction = model.predict(new_img)
            print(prediction)

            if prediction >= 0.5:
                flash('Class A')
            else:
                flash('Class B')

            return redirect(url_for('classifier'))
    return render_template('image_selection.html')

if __name__ == "__main__":
    app.run(debug=True)
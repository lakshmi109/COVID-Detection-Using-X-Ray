from flask import Flask, request, render_template, send_from_directory, url_for
import numpy as np
import pickle
import tensorflow as tf

from PIL import Image

from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
 
app = Flask(__name__) 
app.config['SECRET_KEY'] = 'asjdkhajhd'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')]
    )
    submit = SubmitField('Predict') 


model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET','POST'])
def predict():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        # get the pixel values of the image
        img =  np.array(Image.open(app.config['UPLOADED_PHOTOS_DEST']+'/'+filename).resize((80,80)))/255
        # Convert the image to tensor
        img_tensor = tf.expand_dims(img, axis=0)
        prediction = np.argmax(model.predict(img_tensor), axis = 1)
        return render_template('index.html', form=form, file_url=file_url, prediction = prediction)
    else:
        file_url = None
        return render_template('index.html', form=form, file_url=file_url)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
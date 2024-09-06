from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from tensorflow.keras.preprocessing import image


TEMPLATE_DIR = os.path.abspath('../templates')
STATIC_DIR = os.path.abspath('../static')

app = Flask(__name__)


model = load_model('best_model.h5')
target_img = os.path.join(os.getcwd() , 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224,224,3))
    #img = img.reshape(1,224,224,3)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    return x

@app.route('/index',methods=['GET','POST'])
def fundus_check():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x < 0.5:
                cancer = "Benign"
          
            else:
                cancer = "Malignant"
            #'retina' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('predict.html', cancer = cancer,prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/predict',methods=['GET','POST'])
def predict_check():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x < 0.45:
                cancer = "Benign"
            else:
                cancer = "Malignant"
           
            #'retina' , 'prob' . 'user_image' these names we have seen in predict.html.
            [['{:.2f}'.format(class_prediction) for class_prediction in sublist] for sublist in class_prediction]

            return render_template('predict.html', cancer = cancer,prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)
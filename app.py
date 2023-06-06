import os
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import sys
import cv2
import numpy as np

ROOT_PATH = 'http://ml.anyastunting.com/'
# ROOT_PATH = 'localhost:6969/'

sys.path.append("..")
app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'result/'

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if img is not None:
            images.append(img)
    return images

# Endpoint 
@app.route('/')
def index():
    hello_json = {
        'status_code': 200,
        'message': 'Success testing the API!',
        'data': [],
    }
    return jsonify(hello_json)


@app.route('/post', methods=['POST'])
def post():
    data = request.get_json()
    return jsonify(data)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Load Aruco detector
        parameters = cv2.aruco.DetectorParameters_create()
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        
        # Load Object Detector
        detector = HomogeneousBgDetector()

        # Load Image
        img = cv2.imread('uploads/' + filename)

        # Get Aruco marker
        corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(img)

        height_list = []
        width_list = []


        # Draw objects boundaries
        for i, cnt in enumerate(contours):
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            height_list.append(object_height)
            width_list.append(object_width)

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            
        cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filename), img)

        detected_height = {height_list, width_list}
        json = {
            # 'label': label.replace('_', ' '),
            # 'image_url': 'http://127.0.0.1:6969/result/' + filename
            'image_url': ROOT_PATH + '/result/' + filename,
            'detected_list' : detected_height
        }
        return jsonify(json)
    else:
        json = {
            'data': [],
            'message': 'Please upload JPG, JPEG, or PNG!',
            'error': "The selected file isn't supported!"
        }
        return jsonify(json)


@app.route('/result/<name>')
def output_file(name):
    return send_from_directory(app.config['OUTPUT_FOLDER'], name)


@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Endpoint not found', 'status_code': 404})

def create_app():
   return app

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6969)
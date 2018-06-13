from flask import Flask, render_template, Response
import cv2
import os
import glob

app = Flask(__name__)

def get_frame(image_adr):
    image = cv2.imread(image_adr)
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        list_of_files = glob.glob('../image/*.jpg') # * means all if need specific format then *.csv
        image_adr = max(list_of_files, key=os.path.getctime)
        frame = get_frame(image_adr)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
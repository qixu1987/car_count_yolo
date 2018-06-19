"""server flask to show results on a html page """
from flask import Flask, render_template, Response, request
import cv2
from os.path import dirname,abspath,join

dir_path = dirname(dirname(dirname(abspath(__file__))))
image_add = join(dir_path,"image/frame.jpg.done")
app = Flask(__name__)

# to be refacto
def get_frame(image_add = image_add):
    image = cv2.imread(image_add)
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
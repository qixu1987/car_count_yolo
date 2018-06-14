from flask import Flask, render_template, Response, request
import cv2
import time
import jsonpickle
import numpy as np


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
 #       list_of_files = glob.glob('../image/*.jpg') # * means all if need specific format then *.csv
  #      image_adr = list_of_files[-1]
        image_adr = "../image/frame.jpg.done"
        print (image_adr)
        frame = get_frame(image_adr)
        time.sleep(0.08)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
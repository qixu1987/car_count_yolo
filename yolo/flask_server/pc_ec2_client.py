import requests
import json
import cv2
import time

addr = 'http://darwin-ia.bigdata.digital.engie.com/'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('../image/frame_test.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)

# send http request with image and receive response
for a in range(1000):
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    #time.sleep(0.05)
# decode response
print json.loads(response.text)

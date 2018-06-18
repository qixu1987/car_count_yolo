import cv2
from yolos.yolo3 import *
from yolos.proc import *
import time



url = "rtsp://admin:engie@86.67.73.30:8082/live/ch0"
vidcap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
vidcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


currentFrame = 0
while(True):

    success, image = vidcap.read()
    cv2.imshow('image',image)
    # To stop duplicate images
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()



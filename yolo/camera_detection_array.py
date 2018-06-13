import cv2
import numpy as np
from yolos.yolo3 import *
from yolos.proc_hp import *
import time

# Playing video from file:
#cap = cv2.VideoCapture('../videoplayback.mp4')
# Capturing video from webcam:



weights_path = "../yolov3.weights"

# set some parameters
net_h, net_w = 224, 224
nb_box =3
nb_models = 3
obj_thresh, nms_thresh = 0.5, 0.2
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# make the yolov3 model to predict 80 classes on COCO
yolov3 = make_yolov3_model()

# load the weights trained on COCO into the model
weight_reader = WeightReader(weights_path)
weight_reader.load_weights(yolov3)

### here starts the recode part 
pos_array,raw_col_list = postion_array(net_h, net_w,anchors,nb_box,nb_models)
class_list = None
if class_list is None :
   class_list = labels
class_ind,class_labels = class_to_ind(class_list,labels)


cap = cv2.VideoCapture(0)
currentFrame=0
while(True):
    # Capture frame-by-frame
    count = 1
    ret, frame = cap.read()
    factor = 12
    while not (count % factor):
        ret, frame = cap.read()
        count = count+1
    count = 1  

    # Handles the mirroring of the current frame
    start = time.time()
    image_h, image_w, _ = frame.shape
    new_image = preprocess_input(frame, net_h, net_w)
    
    # run the prediction
    start = time.time()
    yolos = yolov3.predict(new_image)
    print(start - time.time())
      
    start = time.time()
    box_array =  np.array([]).reshape(0,(4+len(class_list)))
    for ind_model in range(nb_models):
        box_array = np.concatenate((box_array,
                                    decode_netout_mat(yolos[ind_model][0],
                                    anchors[ind_model], obj_thresh,
                                    nms_thresh, net_h, net_w,raw_col_list[ind_model],
                                    class_ind,pos_array[ind_model]))
                        )

    new_image_2 = new_image.reshape(net_h,net_w,3)
    if box_array.shape[0]:
        box_array_list = do_nms(box_array, nms_thresh,obj_thresh)
        draw_boxes(new_image_2,box_array_list,class_labels,net_w,net_h)
    # Display the resulting frame
    cv2.imshow('image',cv2.resize(new_image_2, (768, 768)) )
    end = time.time()
    print(start-end)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

image_ind_list = [0]
app.run(host='127.0.0.1', debug=True)

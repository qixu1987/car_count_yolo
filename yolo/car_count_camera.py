import cv2
from yolos.yolo3 import *
from yolos.proc_hp import *
from yolos.zone import *
import time
import os


weights_path = "../yolov3.weights"
image_folder = "../image"

# set some parameters
net_h, net_w = 160, 160
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

### here starts the recode part
pos_array,raw_col_list = postion_array(net_h, net_w,anchors,nb_box,nb_models)
class_list =["person", "bicycle", "car", "motorbike", "bus", "truck"]
if class_list is None :
    class_list = labels
class_ind,class_labels = class_to_ind(class_list,labels)

zone_list =   [Zone(75,30,85,150)]

# make the yolov3 model to predict 80 classes on COCO
yolov3 = make_yolov3_model()

# load the weights trained on COCO into the model
weight_reader = WeightReader(weights_path)
weight_reader.load_weights(yolov3)



url = "rtsp://admin:engie@86.67.73.xx:8082/live/ch0"
vidcap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)

success, image = vidcap.read()
image = image[440:600,780:940,:]
count = 0
success = True
sample_factor = 1
while success:
    print(count)
    start1 = time.time()
    if not (count % sample_factor):
        # Handles the mirroring of the current frame

        image_h, image_w, _ = image.shape
        new_image = preprocess_input(image, net_h, net_w)

        # run the prediction
        start = time.time()
        yolos = yolov3.predict(new_image)
        print(start - time.time())

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
            new_image_2 = count_car(new_image_2,box_array_list,class_labels,net_w,net_h,zone_list)

        cv2.imwrite(image_folder + "/frame.jpg".format(count), new_image_2*255.)
        os.rename(image_folder + "/frame.jpg".format(count),image_folder + "/frame.jpg.done".format(count))
        #nb_buffet = 100
        #if count >= nb_buffet:
        #    os.remove(image_folder + "/frame.jpg".format(count-nb_buffet))


    success,image = vidcap.read()
    image = image[420:580,780:940,:]
    print(start1 - time.time())
    print('Read a new frame: ', success)
    count += 1


import cv2
import numpy as np
from zone import *
from yolo3 import *
import cv2
import time

# Playing video from file:
#cap = cv2.VideoCapture('../videoplayback.mp4')
# Capturing video from webcam:



def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def postion_array(net_h, net_w,nb_box,nb_models):
    """
    generate array headers with row position, col position, anchors box w and 
    anchor h for each of 3 models 
    """
    position_array_list = []
    raw_col_list = []
    grid_pixel = [32,16,8]
    for box_ind in range(nb_models):
        row_nb = net_h/grid_pixel[box_ind]
        col_nb = net_w/grid_pixel[box_ind]
        pos_matrix = np.array([[row,col,anchors[box_ind][2 * b + 0],anchors[box_ind][2 * b + 0]] 
                                    for row in range(row_nb) 
                                    for col in range(col_nb)
                                    for b in range(nb_box)
                                    ])   
        position_array_list.append(pos_matrix)
        raw_col_list.append((row_nb,col_nb))
    return position_array_list, raw_col_list    

def decode_netout_mat(netout, anchors, obj_thresh, nms_thresh, net_h, net_w,
                      raw_col,class_list,pos_matrix):
    """
    decode yolo outputs to arrays with 
    O:row 1:col 2: box_w 3:box_h 4:x 5:y 6:h 7:w 8:obj and classes probability
    """
    row_nb,col_nb = raw_col
    grid_h , grid_w = raw_col
    
    netout = np.array(netout).reshape((row_nb, col_nb, 3, -1)).reshape(row_nb*col_nb*3,85)
    netout=np.concatenate((pos_matrix,netout),axis=1)
    # O:row 1:col 2: box_w 3:box_h 4:x 5:y 6:h 7:w 8:obj
    keep_list = [0,1,2,3,4,5,6,7,8] + class_list
    # drop not useful class
    netout =  netout[ :,keep_list]
    netout[..., 4:6]  = _sigmoid(netout[..., 4:6])
    netout[ :,8:]  = _sigmoid(netout[:,8:])
    # drop no object detection lines
    netout = netout[netout[:, 8]>obj_thresh,:]
    # prob for each class
    netout[:, 9:]  = netout[:,8][:, np.newaxis] * netout[:, 9:]
    
    # calculate x, y, h, w,
    netout[:,4] =  (netout[:,4] + netout[:,1])/ grid_w
    netout[:,5] =  (netout[:,5] + netout[:,0])/ grid_w
    netout[:, 7]= netout[:, 2]* np.exp(netout[:, 7])/net_w
    netout[:, 6]= netout[:, 3]* np.exp(netout[:, 6])/net_h
    
    #calculate  xmin, ymin, xmax, ymax and concatenation
    if netout[:,9:].shape>1:
        netout=np.concatenate( 
                ((netout[:,4]- 0.5*netout[:, 7]).reshape(-1,1),(netout[:,5]- 0.5*netout[:, 6]).reshape(-1,1),
                (netout[:,4]+ 0.5*netout[:, 7]).reshape(-1,1),(netout[:,5]+ 0.5*netout[:, 6]).reshape(-1,1),
                 netout[:,9:]),axis=1
         )
    else:
        netout=np.concatenate( 
                ((netout[:,4]- 0.5*netout[:, 7]).reshape(-1,1),(netout[:,5]- 0.5*netout[:, 6]).reshape(-1,1),
                (netout[:,4]+ 0.5*netout[:, 7]).reshape(-1,1),(netout[:,5]+ 0.5*netout[:, 6]).reshape(-1,1),
                 netout[:,9:].reshape(-1,1)),axis=1
         )
    return netout        

def class_to_ind(class_list,labels):
    """
    the col number of classes in array
    """
    base_num = 9
    # a refactorer
    class_ind= [i+base_num for i in range(len(labels))
                if labels[i] in class_list]
    class_labels = [labels[i] for i in range(len(labels))
                if labels[i] in class_list]     
    return class_ind,class_labels

def box_in_image(anchor,net_h,net_w):
    if net_w > max([anchor[i] for i in [0,2,4]]) and net_h > max([anchor[i] for i in [1,3,5]]):
        return True
    else:
        return False


def draw_boxes(image,box_array,labels,net_w,net_h):
    for ind_class in range(len(labels)):
        box_array_class =  box_array[ind_class]
        if box_array_class.shape[0]: 
            for ind_col in range(box_array_class.shape[0]):
                cv2.rectangle(image, (int(box_array_class[ind_col,0]*net_w),
                                      int(box_array_class[ind_col,1]*net_h)),
                                     (int(box_array_class[ind_col,2]*net_w),
                                      int(box_array_class[ind_col,3]*net_h)), 
                                      (0,255,0), 1)
                cv2.putText(image, 
                            labels[ind_class] + ' ' + "{:.3f}".format(box_array_class[ind_col,4]*100) + '%', 
                            (int(box_array_class[ind_col,0]*net_w), 
                                 int(box_array_class[ind_col,1]*net_h) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            2e-3 * image.shape[0],
                            (0,255,0), 1)
        else:
            continue       
    return image 


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box_array,row1, row2):
    intersect_w = _interval_overlap([box_array[row1,0], box_array[row1,2]], 
                                    [box_array[row2,0], box_array[row2,2]])
    
    intersect_h = _interval_overlap([box_array[row1,1], box_array[row1,3]], 
                                    [box_array[row2,1], box_array[row2,3]])

    intersect = intersect_w * intersect_h

    w1, h1 = box_array[row1,2]-box_array[row1,0], box_array[row1,3]-box_array[row1,1]
    w2, h2 = box_array[row2,2]-box_array[row2,0], box_array[row2,3]-box_array[row2,1]

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union


def do_nms(box_array, nms_thresh,obj_thresh):
    box_array_class = []
    if box_array.shape[0] > 0:
        for c in range(4,box_array.shape[1]):
            box_class_array = box_array[:,[0,1,2,3]+[c]]
            box_class_array = box_class_array[box_class_array[:,4]>obj_thresh,:]
            if  not box_class_array.shape[0]:
                box_array_class.append(np.array([])) 
            else: 
                delete_list=[]
                sorted_indices = np.argsort(box_class_array[:,4])
                for i in range(len(sorted_indices)):
                    row1 = sorted_indices[i]       
                    if row1 in delete_list: continue        
                    for j in range(i+1, len(sorted_indices)):
                        row2 = sorted_indices[j]
                        if bbox_iou(box_class_array,row1, row2) >= nms_thresh:
                            delete_list.append(row2)
                if len(delete_list):
                    box_array_class.append( np.delete(box_class_array,sorted_indices[delete_list],0))
                else:
                    box_array_class.append( box_class_array)
    return box_array_class    

   

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
pos_array,raw_col_list = postion_array(net_h, net_w,nb_box,nb_models)
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
    cv2.imwrite("test.jpg",new_image_2)
    cv2.imwrite("test2.jpg",new_image_2*255.)
    end = time.time()
    print(start-end)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



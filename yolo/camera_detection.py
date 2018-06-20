"""real time webcame yolo3 object detection"""
import argparse
import cv2
from .yolos.yolo3 import *
from .yolos.proc_hp import *
from .utils.json_process import load_json
import time


argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to weights file')

argparser.add_argument(
    '-j',
    '--json',
    help='json config address')


def _main_(args):
    # set some parameters
    weights_path = args.weights
    json_name = args.json
    param_dic = load_json(json_name)
    # set some parameters see yolos doc for more information
    net_h, net_w = param_dic["net_h"], param_dic["net_w"]
    obj_thresh, nms_thresh = param_dic["obj_thresh"], param_dic["nms_thresh"]
    anchors = param_dic["anchors"]
    labels = param_dic["labels"]
    nb_models = param_dic["nb_models"]
    nb_box = param_dic["nb_box"]
    factor = param_dic["factor_sampling"]

    # make the yolov3 model to predict 80 classes on COCO
    yolov3 = make_yolov3_model()
    # load the weights trained on COCO into the model
    weight_reader = WeightReader(weights_path)
    weight_reader.load_weights(yolov3)

    # creation position array list and class list
    pos_array,raw_col_list = postion_array(net_h, net_w,anchors,nb_box,nb_models)
    class_list = None
    if class_list is None :
       class_list = labels
    class_ind,class_labels = class_to_ind(class_list,labels)

    # open the camera
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        count = 1
        ret, frame = cap.read()
        while not (count % factor):
            ret, frame = cap.read()
            count = count+1
        count = 1

        image_h, image_w, _ = frame.shape
        # image preprocessing
        new_image = preprocess_input(frame, net_h, net_w)

        # run the prediction
        start = time.time()
        yolos = yolov3.predict(new_image)
        print(start - time.time())

        start = time.time()
        box_array =  np.array([]).reshape(0,(4+len(class_list)))
        for ind_model in range(nb_models):
            box_array = np.concatenate((box_array,
                                        decode_netout_mat(yolos[ind_model][0], obj_thresh,
                                        net_h, net_w,raw_col_list[ind_model],
                                        class_ind,pos_array[ind_model]))
                            )
        # resize image for drawing
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

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
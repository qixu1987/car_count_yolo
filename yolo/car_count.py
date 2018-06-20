"""car count use yolo3 example
   2 zones are set between the  116:308,116:308 pixels in the video sample
   to connect it to web cam put the flow link after the -v argument
"""
import argparse
import cv2
from .yolos.yolo3 import *
from .yolos.proc_hp import *
from .yolos.zone import *
from .utils.json_process import load_json
import time
import os

argparser = argparse.ArgumentParser(
    description='car count use yolo3')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to weights file')

argparser.add_argument(
    '-i',
    '--image',
    help='path to image folder')

argparser.add_argument(
    '-v',
    '--video',
    help='path to video file')

argparser.add_argument(
    '-u',
    '--url',
    help='url of video flow')

argparser.add_argument(
    '-j',
    '--json',
    help='json config address')


def _main_(args):
    weights_path = args.weights
    image_folder = args.image
    video_path = args.video
    url = args.url
    json_name = args.json

    # set some parameters
    param_dic = load_json(json_name)
    # set some parameters see yolos doc for more information
    net_h, net_w = param_dic["net_h"], param_dic["net_w"]
    obj_thresh, nms_thresh = param_dic["obj_thresh"], param_dic["nms_thresh"]
    anchors = param_dic["anchors"]
    labels = param_dic["labels"]
    nb_models = param_dic["nb_models"]
    nb_box = param_dic["nb_box"]
    sample_factor = param_dic["factor_sampling"]
    class_list = param_dic["detection_class"]
    detect_zone = param_dic["detection_zone"]

    ### here starts the recode part
    pos_array,raw_col_list = postion_array(net_h, net_w,anchors,nb_box,nb_models)
    if class_list is None :
        class_list = labels
    class_ind,class_labels = class_to_ind(class_list,labels)
    zone_list =  [Zone(*pos) for pos in param_dic["count_zone_config"]]

    # make the yolov3 model to predict 80 classes on COCO
    yolov3 = make_yolov3_model()

    # load the weights trained on COCO into the model
    weight_reader = WeightReader(weights_path)
    weight_reader.load_weights(yolov3)
    if url is None:
        vidcap = cv2.VideoCapture(video_path)
    else:
        vidcap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
        vidcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    success, image = vidcap.read()
    image = image[detect_zone[0]:detect_zone[1],detect_zone[0]:detect_zone[1],:]
    count = 0
    success = True
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
                                            decode_netout_mat(yolos[ind_model][0], obj_thresh,
                                                              net_h, net_w,raw_col_list[ind_model],
                                                              class_ind,pos_array[ind_model]))
                                           )

            new_image_2 = new_image.reshape(net_h,net_w,3)
            if box_array.shape[0]:
                box_array_list = do_nms(box_array, nms_thresh,obj_thresh)
                new_image_2 = count_car(new_image_2,box_array_list,class_labels,net_w,net_h,zone_list)

            cv2.imwrite(image_folder + "/frame.jpg".format(count), new_image_2*255.)
            os.rename(image_folder + "/frame.jpg".format(count),image_folder + "/frame.jpg.done".format(count))

        success,image = vidcap.read()
        image = image[detect_zone[0]:detect_zone[1],detect_zone[0]:detect_zone[1],:]
        print(start1 - time.time())
        print('Read a new frame: ', success)
        count += 1

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
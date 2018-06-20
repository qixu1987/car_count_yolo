import json
import os

class_labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                "banana","apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
                "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

# camera detection
camera_json = {  "labels":class_labels,
                 "anchors":anchors,
                 "net_h": 224,
                 "net_w": 224,
                 "obj_thresh": 0.5,
                 "nms_thresh": 0.2,
                 "factor_sampling":3,
                 "nb_box":3,
                 "nb_models":3,
     }

# video car count
video_car_json = {  "labels":class_labels,
                 "anchors":anchors,
                 "net_h": 192,
                 "net_w": 192,
                 "obj_thresh": 0.5,
                 "nms_thresh": 0.2,
                 "factor_sampling":1,
                 "nb_box":3,
                 "nb_models":3,
                 "count_zone_config":[(15,125,85,135),(85,125,155,135)],
                 "detection_class": ["person", "bicycle", "car", "motorbike", "bus", "truck"],
                 "detection_zone":(116,308,116,308)
                 }

def load_json(json_name):
    json_add = os.path.join(json_name)
    f = open(json_add)
    param_dic = json.load(f)
    f.close()
    return param_dic

def make_json(json_name,json_dic):
    json_add = os.path.join('../../seed',json_name)
    with open(json_add, 'w') as fp:
        json.dump(json_dic, fp)


if __name__ == '__main__':
    # camera detection
    make_json("webcam.json",camera_json)
    make_json("video_car_count.json",video_car_json)
import argparse
import os
from zone import *
from yolo3 import *
import cv2
import time


np.set_printoptions(threshold=np.nan)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#  parsing of arguments:
argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to weights file')

argparser.add_argument(
    '-v',
    '--video',
    help='path to video file')


def draw_boxes(image, boxes, labels, obj_thresh,center_zone):
    centre_zone_active = [False]*len(center_zone)
    for box in boxes:
        label_str = ''
        label = -1
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                print(labels[i] + ': ' + "{:.3f}".format(box.classes[i]*100) + '%'  )

        if label_str in ["bicycle", "car","motorbike", "bus", "truck"]:
            in_center_zone = False

            for zone_index in range(len(center_zone)):
                if center_zone[zone_index].center_in_zone(box):
                    cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (255,0,0), 3)
                    centre_zone_active[zone_index] = True
                    in_center_zone = True

            if not in_center_zone:
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)

            cv2.putText(image,
                        label_str + ' ' + "{:.3f}".format(box.get_score()) + '%' ,
                        (box.xmin, box.ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0,255,0), 2)

    for zone_index in range(len(center_zone)):
        zone = center_zone[zone_index]
        zone.count_increment(centre_zone_active[zone_index])
        cv2.rectangle(image, (zone.xmin,zone.ymin), (zone.xmax,zone.ymax), (255,0,0), 3)
        cv2.putText(image,
                    str(zone.count),
                    (zone.xmin, zone.ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2e-3 * image.shape[0],
                    (255,0,0), 2)

    cv2.putText(image,
                "M2 Car nb: "+ str(sum([zone.count for zone in center_zone])),
                #(100, 250),
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.003 * image.shape[0],
                (0,255,0), 2)

    return image

# processing the image
def image_processing(image,count,image_folder,net_h,
                     net_w,anchors, obj_thresh, nms_thresh,labels,yolov3,center_zone):
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    # run the prediction
    yolos = yolov3.predict(new_image)
    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)

    # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, labels, obj_thresh,center_zone)

    # write the image with bounding boxes to file
    cv2.imwrite(image_folder + "/frame%d.jpg" % count, image)


def _main_(args):
    weights_path = args.weights
    video_path   = args.video

    # set some parameters
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.3, 0.5
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

    image_folder = "image"

    sig = 5
    c_y = 1
    c_x = 4
    center_zone = [Zone(200-c_y*sig,200+c_y*sig,225-c_x*sig,225+c_x*sig),
                   Zone(200-c_y*sig,200+c_y*sig,265-c_x*sig,265+c_x*sig),
                   Zone(200-c_y*sig,200+c_y*sig,370-c_x*sig,370+c_x*sig),
                   Zone(200-c_y*sig,200+c_y*sig,410-c_x*sig,410+c_x*sig),]

    # make the yolov3 model to predict 80 classes on COCO
    yolov3 = make_yolov3_model()

    # load the weights trained on COCO into the model
    weight_reader = WeightReader(weights_path)
    weight_reader.load_weights(yolov3)

    # image processing pipeline
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    sample_factor = 1
    while success:
        print(count)
        if not (count % sample_factor):
            start_time = time.time()
            image_processing(image,count,image_folder,net_h,
                             net_w,anchors, obj_thresh, nms_thresh,labels,yolov3,center_zone)
            end_time = time.time()
            print(end_time-start_time)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    video_name = 'car_count.avi'

    images = {int(img.split('.')[0][5:]):img for img in os.listdir(image_folder) if img.endswith(".jpg")}
    index=sorted(images.keys())
    frame = cv2.imread(os.path.join(image_folder, images[index[0]]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, 24/sample_factor, (width,height))
    for image_ind in index:
        video.write(cv2.imread(os.path.join(image_folder, images[image_ind])))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

"""array implementation of yolo pre/post processing"""
import cv2
import numpy as np

def preprocess_input(image, net_h, net_w):
    """ pre processing of image
    :param image:
    :param net_h: height of yolo input image
    :param net_w: width of yolo input image
    :return: image
    """
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)/new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)/new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2),
             int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def postion_array(net_h, net_w,anchors,nb_box,nb_models):
    """generate arrays with row position, col position, anchors box width and
    anchor h for each of the models to make it possible for array implementation
    :param net_h: height of yolo input image
    :param net_w: width of yolo input image
    :param anchors: list of anchors box dimensions
    :param nb_box:  nb of anchors box in each model (3)
    :param nb_models: nb or models used by yolo (3)
    :return: list of position arrays
    """
    position_array_list = []
    raw_col_list = []
    grid_pixel = [32,16,8] # see yolov3 documentation
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


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout_mat(netout, obj_thresh, net_h, net_w,
                      raw_col,class_list,pos_matrix):
    """transform and select yolo outputs to array
    :param netout:  list of yolo model out put each element is a result of one anchors box
    :param obj_thresh: object probabilities threshold
    :param net_h: height of yolo input image
    :param net_w: width of yolo input image
    :param raw_col: number of grid
    :param class_list: position of selected classes
    :param pos_matrix: see postion_array function
    :return: array of xmin, ymin, xmax, ymax and classes probability for each anchor box
    """
    row_nb,col_nb = raw_col
    grid_h , grid_w = raw_col

    # convert fo array
    netout = np.array(netout).reshape((row_nb, col_nb, 3, -1)).reshape(row_nb*col_nb*3,85)
    netout=np.concatenate((pos_matrix,netout),axis=1)

    # O:row 1:col 2: box_w 3:box_h 4:x 5:y 6:h 7:w 8:obj
    keep_list = [0,1,2,3,4,5,6,7,8] + class_list
    # drop not useful class
    netout =  netout[ :,keep_list]
    #classes and obj probabilities
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
            ((netout[:,4]- 0.5*netout[:, 7]).reshape(-1,1),
             (netout[:,5]- 0.5*netout[:, 6]).reshape(-1,1),
             (netout[:,4]+ 0.5*netout[:, 7]).reshape(-1,1),
             (netout[:,5]+ 0.5*netout[:, 6]).reshape(-1,1),
             netout[:,9:]),axis=1
        )
    else:
        # reshape if there is less than one detection (np array shape problem for dimension 1)
        netout=np.concatenate(
            ((netout[:,4]- 0.5*netout[:, 7]).reshape(-1,1),
             (netout[:,5]- 0.5*netout[:, 6]).reshape(-1,1),
             (netout[:,4]+ 0.5*netout[:, 7]).reshape(-1,1),
             (netout[:,5]+ 0.5*netout[:, 6]).reshape(-1,1),
             netout[:,9:].reshape(-1,1)),axis=1
        )
    return netout

def class_to_ind(class_list,labels):
    """ col number of classes in array
    :param class_list: interested classes in english
    :param labels: all the classes of model in english
    :return: position of interested classes in array and there label in english
    """
    base_num = 9 # first col of class probability
    class_ind= [i+base_num for i in range(len(labels))
                if labels[i] in class_list]
    class_labels = [labels[i] for i in range(len(labels))
                    if labels[i] in class_list]
    return class_ind,class_labels

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
    """ calculate the IOU of 2 anchors box in array
    :param box_array: array of anchor boxes detection
    :param row1: row of first anchor box
    :param row2: row of second anchor box
    :return: intersection of union rate
    """
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
    """ Non-Maximum Suppression
    :param box_array: see decode_netout_mat function
    :param nms_thresh: Non-Maximum Suppression threshold
    :param obj_thresh: obj probability threshold
    :return: list of filtered anchor box array by class
    """
    box_array_class = []
    first_class_prob = 4
    if box_array.shape[0] > 0:
        for c in range(first_class_prob,box_array.shape[1]):
            box_class_array = box_array[:,[0,1,2,3]+[c]]
            box_class_array = box_class_array[box_class_array[:,4]>obj_thresh,:]
            if  not box_class_array.shape[0]:
                box_array_class.append(np.array([]))
            else:
                delete_list=[]
                # sort by class probabilities
                sorted_indices = np.argsort(box_class_array[:,4])[::-1]
                for i in range(len(sorted_indices)):
                    row1 = sorted_indices[i]
                    if row1 in delete_list: continue
                    for j in range(i+1, len(sorted_indices)):
                        row2 = sorted_indices[j]
                        if bbox_iou(box_class_array,row1, row2) >= nms_thresh:
                            # delete the anchors boxes which has lower probabilities
                            # and an IOU above the threshold
                            delete_list.append(row2)
                if len(delete_list):
                    box_array_class.append( np.delete(box_class_array,delete_list,0))
                else:
                    box_array_class.append( box_class_array)
    return box_array_class

def count_car(image,box_array,labels,net_w,net_h,zone_list):
    """ draw vehicle anchors box and count
    :param image:
    :param box_array: see do_nms function
    :param labels: list of classes name
    :param net_w: width of yolo input image
    :param net_h: height of yolo input image
    :param zone_list: list of zone instances
    :return:
    """
    zone_stat = [[False,zone] for zone in zone_list] # [occupation,zone]

    if box_array:
        for ind_class in range(len(labels)):
            box_array_class =  box_array[ind_class]
            if box_array_class.shape[0]:
                for ind_col in range(box_array_class.shape[0]):
                    # conversion to image pixel position
                    pos_array = box_array_class[ind_col,0:4]
                    pos_array[[0,2]] = pos_array[[0,2]] * net_w
                    pos_array[[1,3]] = pos_array[[1,3]] * net_h

                    for zone_index in range(len(zone_stat)):
                        if zone_stat[zone_index][1].center_in_zone_array(pos_array):
                            # set occupation as True if the box center contained in zone
                            zone_stat[zone_index][0] = True
                    # draw box
                    cv2.rectangle(image, (int(pos_array[0]),
                                          int(pos_array[1])),
                                  (int(pos_array[2]),
                                   int(pos_array[3])),
                                  (0,255,0), 1)
                    # draw class name and probabilities
                    cv2.putText(image,
                                labels[ind_class] + ' ' + "{:.3f}".format(box_array_class[ind_col,4]*100) + '%',
                                (int(pos_array[0]),
                                 int(pos_array[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2e-3 * image.shape[0],
                                (0,255,0), 1)
            else:
                continue
    # for each zone calculate incrementation and draw zone box and count
    for zone_index in range(len(zone_stat)):
        zone = zone_stat[zone_index][1]
        zone.count_increment(zone_stat[zone_index][0])
        cv2.rectangle(image, (zone.xmin,zone.ymin), (zone.xmax,zone.ymax), (255,0,0), 1)

        cv2.putText(image,
                        str(zone.count),
                        (zone.xmin, zone.ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2e-3 * image.shape[0],
                        (255,0,0), 2)
    # draw total car number
    cv2.putText(image,
                "M2 Car nb: "+ str(sum([zone[1].count for zone in zone_stat])),
                #(100, 250),
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.002 * image.shape[0],
                (0,255,0), 2)
    return image



def draw_boxes(image,box_array,labels,net_w,net_h):
    """ draw object detected anchors box on image
    :param image:
    :param box_array: see do_nms function
    :param labels: list of object names
    :param net_w: width of yolo input image
    :param net_h: height of yolo input image
    :return: image
    """
    for ind_class in range(len(labels)):
        box_array_class =  box_array[ind_class]
        if box_array_class.shape[0]:
            for ind_col in range(box_array_class.shape[0]):
                # draw anchors box
                cv2.rectangle(image, (int(box_array_class[ind_col,0]*net_w),
                                      int(box_array_class[ind_col,1]*net_h)),
                              (int(box_array_class[ind_col,2]*net_w),
                               int(box_array_class[ind_col,3]*net_h)),
                              (0,255,0), 1)
                # draw class name and probability
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
import cv2
import numpy as np

def preprocess_input(image, net_h, net_w):
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
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def postion_array(net_h, net_w,anchors,nb_box,nb_models):
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


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:10:34 2018

@author: ia2069

"""
pos_matrix = np.array([[row,col,anchors[b_group][2 * b + 0],anchors[b_group][2 * b + 0]] 
                                    for row in range(row_nb) 
                                    for col in range(col_nb)
                                    for b in range(nb_box)
                                    ])
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
b_group=2
b=2
# recode
netout = yolos[2][0]

def decode_netout_mat(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
    start =  time.time()
    row_nb = col_nb = 32
    netout = np.array(netout).reshape((row_nb, col_nb, 3, -1)).reshape(row_nb*col_nb*3,85)
    netout=np.concatenate((pos_matrix,netout),axis=1)
    class_list = [11,14,16]
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
    
    # to be set as parameters
    grid_w=32
    grid_h = 32
    
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
   
    print(time.time()-start)
    return netout
    


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

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
    


    


b=time.time() 
a=decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w)

print(time.time()-b)


c=time.time() 
do_nms(b, nms_thresh,obj_thresh)
print(time.time()-c)
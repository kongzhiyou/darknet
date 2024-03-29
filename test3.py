from ctypes import *
import random
import os
import math
from Tracker import add_track
import cv2
import time
import numpy as np
import glob
from Tracker import update_track


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE


def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image


# def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    # num = c_int(0)
    # pnum = pointer(num)
    # im = load_image(image.encode(), 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                #标签名称、置信度、（中心点x，中心点y,边框的宽，边框的高）
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def track_dect(track_list,c_x,c_y):
    min = 10000
    for trac in track_list:
        trac.flag = 0
        if(trac.care == 1):
            x = math.fabs(trac.x-c_x)
            y = math.fabs(trac.y-c_y)
            value = math.sqrt(x*x+y*y)
            if value<min:
                min = value
                target = trac.id+1
    if (min > 30):
        target = None
    return target,track_list


if __name__ == "__main__":

    net = load_net("cfg/label.cfg".encode(), "backup/label_2000.weights".encode(), 0)
    meta = load_meta("data/label.data".encode())

    vedio_list = glob.glob('/Users/peter/anji/anjicaiji_8/round5_1_2.avi')
    for vedio_path in vedio_list:
        print(vedio_path)
        # tracker = Tracker(100, 8, 15, 0)
        cap = cv2.VideoCapture(vedio_path)
        frames_num = cap.get(7)
        if cap.isOpened():
            success = True
            print('video open successfully')
        else:
            success = False
            print('video open failed')


        i = 0
        interval = 1
        img_list = []

        fps = 24
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(vedio_path[:-4] + '_test_2.avi', fourcc, fps,
                                      (2048, 2448))  # 最后一个是保存图片的尺寸
        if not os.path.exists('video_save'):
            os.makedirs('video_save')

        bbox_hist_list = []
        package_num = []
        door_district_x_range = [150, 400]
        door_district_x_range_narrow = [210, 350]
        in_door_count = 0
        temp = None
        last_package_index = 0
        video_image = []
        forklift_temp = ''
        package_name = ''
        package_num_temp = ''
        video_num = 2
        start = time.time()
        print(frames_num)
        tracker_list = []
        while success:
            success, image = cap.read()
            # image =cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # print(image.shape)
            # image = np.rot90(image, -1)
            # print(image.shape)
            if success and i % interval == 0:
                image = np.rot90(image, -1)
                im = nparray_to_image(image)
                res = detect(net, meta, im)
                image = image.copy()
                boxs = []
                number = len(res)
                count = 0
                j = 0
                for cls, score, bbox in res:
                    print('step: ')
                    print(bbox)
                    if((j+1)>len(tracker_list)):
                        track = add_track(bbox[0],bbox[1],count)
                        track.id = count
                        tracker_list.append(track)
                        count += 1
                        target = count
                    else:
                        target,tracker_list = track_dect(tracker_list,bbox[0],bbox[1])
                        if target==None:
                            track = add_track(bbox[0],bbox[1],count)
                            count += 1
                            target = count
                    cls = cls.decode()
                    bb = (bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                    if cls == 'safe_cloth':
                        cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
                    else:
                        cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
                    print(target)
                    cv2.putText(image, cls+str(target-1), (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    for t in tracker_list:
                        if t.id == target-1:
                            t = update_track(t,bbox[0],bbox[1])
                    j+=1
                print('update')

                for t in range(len(tracker_list)):
                    print(tracker_list[t].x, tracker_list[t].y, tracker_list[t].flag)
                    if tracker_list[t].flag == 0:
                        tracker_list[t].care = 0
                videoWriter.write(image)
            i += 1
        end = time.time()
        videoWriter.release()
        end = time.time()
        print((end - start), 'Seconds to finish')



from ctypes import *
import random
import os

import http.client
import urllib.request
import urllib.parse
import urllib.error
import base64
import json
# from aip import AipOcr
import base64
import re
from PIL import Image
from io import BytesIO
from timeit import default_timer as timer
from keras import backend as K
from PIL import ImageFont,ImageDraw
import colorsys
import math
from Tracker import add_track

# package_dict = {'yellow_package': '黄色大包', 'blue_package': '蓝色大包', 'orange_package': '橙色大包',
#                 'white_bag': '白色长包', 'white_package': '白色大包', 'full_tray': '小包满拖', 'half_tray': '小包半拖'}

print(os.getcwd())
print(os.listdir())


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


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


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


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


def calc_center(out_boxes, out_classes, out_scores, score_limit=0.5):
    outboxes_filter = []
    for x, y, z in zip(out_boxes, out_classes, out_scores):
        if z > score_limit:
            if y == 0:  # person标签为0
                outboxes_filter.append(x)

    centers = []
    number = len(outboxes_filter)
    for box in outboxes_filter:
        top, left, bottom, right = box
        center = np.array([[(left + right) // 2], [(top + bottom) // 2]])
        centers.append(center)
    return centers, number

def detect_image(self, image):
    start = timer()

    if self.model_image_size != (None, None):
        assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = self.sess.run(
        [self.boxes, self.scores, self.classes],
        feed_dict={
            self.yolo_model.input: image_data,
            self.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = self.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=self.colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=self.colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    end = timer()
    print(end - start)
    return image,out_boxes, out_scores, out_classes

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    #colors = [(255,99,71) if c==(255,0,0) else c for c in colors ]  # 单独修正颜色，可去除
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors

def trackerDetection(tracker, image, centers, number, max_point_distance=30, max_colors=20, track_id_size=0.8):
    '''
        - max_point_distance为两个点之间的欧式距离不能超过30
            - 有多条轨迹,tracker.tracks;
            - 每条轨迹有多个点,tracker.tracks[i].trace
        - max_colors,最大颜色数量
        - track_id_size,每个中心点，显示的标记字体大小
    '''
    # track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    #            (0, 255, 255), (255, 0, 255), (255, 127, 255),
    #            (127, 0, 255), (127, 0, 127)]
    track_colors = get_colors_for_classes(max_colors)

    result = np.asarray(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, str(number), (20, 40), font, 1, (0, 0, 255), 5)  # 左上角，人数计数

    if (len(centers) > 0):
        # Track object using Kalman Filter
        tracker.Update(centers)
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            # 多个轨迹
            if (len(tracker.tracks[i].trace) > 1):
                x0, y0 = tracker.tracks[i].trace[-1][0][0], tracker.tracks[i].trace[-1][1][0]
                cv2.putText(result, str(tracker.tracks[i].track_id), (int(x0), int(y0)), font, track_id_size,
                            (255, 255, 255), 4)
                # (image,text,(x,y),font,size,color,粗细)
                for j in range(len(tracker.tracks[i].trace) - 1):
                    # 每条轨迹的每个点
                    # Draw trace line
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j + 1][0][0]
                    y2 = tracker.tracks[i].trace[j + 1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    if distance < max_point_distance:
                        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 4)
    return tracker, result

def track_dect(track_list,c_x,c_y):
    if(len(track_list)==0):
        track = add_track(c_x,c_y)
        track_list.append(track)
        target = 0
        return target,track_list
    min = 10000
    i = 0
    for trac in track_list:
        print(trac.x,trac.y)
        x = math.fabs(trac.x-c_x)
        y = math.fabs(trac.y-c_y)
        value = math.sqrt(x*x+y*y)
        if value<min:
            min = value
            target = i
        if(min>30):
            target = None
        i+=1

    return target,track_list


if __name__ == "__main__":
    # net = load_net("cfg/package.cfg".encode(), "backup/new/package_6000.weights".encode(), 0)
    # meta = load_meta("data/package.data".encode())
    net = load_net("cfg/label.cfg".encode(), "backup/label_2000.weights".encode(), 0)
    meta = load_meta("data/label.data".encode())

    from trackerDetection import trackerDetection
    import cv2
    import time
    import numpy as np
    import glob
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    from objecttracker.KalmanFilterTracker import Tracker
    from Tracker import update_track

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
            if success and i % interval == 0:
                image = np.rot90(image, -1)
                im = nparray_to_image(image)
                res = detect(net, meta, im)
                image = image.copy()
                print(res)
                boxs = []
                number = len(res)
                count = 0
                j = 0
                for cls, score, bbox in res:
                    print(bbox)
                    if((j+1)>len(tracker_list)):
                        track = add_track(bbox[0],bbox[1],j)
                        track.id = j
                        tracker_list.append(track)
                        count += 1
                        print(tracker_list[j].x,tracker_list[j].y)
                        target = count
                    else:
                        target,tracker_list = track_dect(tracker_list,bbox[0],bbox[1])
                        if target==None:
                            track = add_track(bbox[0],bbox[1],count)
                            count += 1
                            target = count
                        print(target)
                    cls = cls.decode()
                    bb = (bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                    if cls == 'safe_cloth':
                        cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
                    else:
                        cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
                    print(target)
                    cv2.putText(image, cls+str(target), (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    tracker_list[target-1] = update_track(tracker_list[target-1],bbox[0],bbox[1])
                    print('update')
                    for t in tracker_list:
                        print(t.x,t.y)
                    j+=1

                videoWriter.write(image)
            i += 1
        end = time.time()
        videoWriter.release()
        end = time.time()
        print((end - start), 'Seconds to finish')



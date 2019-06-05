# -*- coding:utf-8 -*-
import os
from ctypes import *
import math
import random
import glob
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
from xml.dom import minidom


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


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image.encode('utf-8'), 0, 0)
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
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def create_xmlFile(annotation_path,image_name,message):

    xml_path = image_name.split('.')[0]+'.xml'
    img = image_name
    print('img: '+img)
    image = cv2.imread(img)

    if not os.path.exists(xml_path):
        doc = Document()
        root = doc.createElement('annotation')
        folder = doc.createElement('folder')
        folder_text = doc.createTextNode(str('add'))
        folder.appendChild(folder_text)
        file_name = doc.createElement('file_name')
        file_name_text = doc.createTextNode(str(message[0]))
        file_name.appendChild(file_name_text)
        root.appendChild(folder)
        root.appendChild(file_name)
        file_path = doc.createElement('path')
        file_path_text = doc.createTextNode(str(image_name))
        file_path.appendChild(file_path_text)
        source = doc.createElement('source')
        database = doc.createElement('database')
        database_text = doc.createTextNode('Unknown')
        database.appendChild(database_text)
        source.appendChild(database)
        segmented = doc.createElement('segmented')
        segmented_text = doc.createTextNode(str(0))
        segmented.appendChild(segmented_text)
        root.appendChild(file_path)
        root.appendChild(source)
        size = doc.createElement('size')
        width = doc.createElement('width')
        height = doc.createElement('height')
        depth = doc.createElement('depth')
        width_text = doc.createTextNode(str(image.shape[1]))
        width.appendChild(width_text)
        height_text = doc.createTextNode(str(image.shape[0]))
        height.appendChild(height_text)
        depth_text = doc.createTextNode(str(image.shape[2]))
        depth.appendChild(depth_text)
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        root.appendChild(size)
        root.appendChild(segmented)
        doc.appendChild(root)
        f = open(xml_path, 'w')
        doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    node_object = ET.Element('object')
    name = ET.Element('name')
    name.text = message[len(message)-1]
    pose = ET.Element('pose')
    pose.text = 'Unspecified'
    node_object.append(name)
    node_object.append(pose)
    truncated = ET.Element('truncated')
    truncated.text = str(0)
    node_object.append(truncated)
    difficult = ET.Element('difficult')
    difficult.text = str(0)
    node_object.append(difficult)
    bndbox = ET.Element('bndbox')
    xmin = ET.Element('xmin')
    xmin.text = str(message[1])
    ymin = ET.Element('ymin')
    ymin.text = str(message[2])
    xmax = ET.Element('xmax')
    xmax.text = str(message[3])
    ymax = ET.Element('ymax')
    ymax.text = str(message[4])
    bndbox.append(xmin)
    bndbox.append(xmax)
    bndbox.append(ymax)
    bndbox.append(ymin)
    node_object.append(bndbox)
    root.append(node_object)
    #tree.write(xml_path,encoding='utf-8',xml_declaration=True)
    xml_string = ET.tostring(root)
    xml_write = minidom.parseString(xml_string)
    with open(xml_path, 'w') as handle:
        xml_write.writexml(handle, indent='\t',newl='\n', addindent='\t',encoding='utf-8')


if __name__ == "__main__":
    # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    # im = load_image("data/wolf.jpg", 0, 0)
    # meta = load_meta("cfg/imagenet1k.data")
    # r = classify(net, meta, im)
    # print r[:10]
    image_path = '/Users/peter/Desktop/video/add'
    save_path = '/Users/peter/Desktop/video/add/result'
    xml_path = '/Users/peter/Desktop/video/add/XML'
    image_list = glob.glob(image_path+'/*.jpg')
    net = load_net("cfg/yuxing_320.cfg".encode(), "backup/yuxing_320_16000.weights".encode(), 0)
    meta = load_meta("data/yuxing_320.data".encode())
    for image in image_list:
        img = cv2.imread(image)
        res= detect(net, meta,image)
        number = len(res)
        # image = img.copy()
        for cls, score, bbox in res:
            cls = cls.decode()
            bb = [int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2), int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]
            print(bb)
            if cls == 'safe_cloth':
                cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
            cv2.putText(img, cls + str(score), (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.imwrite(save_path+'/'+image.split('.')[0].split('/')[-1]+'_result.jpg',img)

            x1, y1, x2, y2 = bb
            class_name = cls
            # try:
            image_name = image.split('.')[0].split('/')[-1]
            msg = [image_name, str(bb[0]), str(bb[1]), str(bb[2]), str(bb[3]),cls]
            print(msg)
            create_xmlFile(xml_path,image, msg)
            # except Exception as e:
            #     print(e)



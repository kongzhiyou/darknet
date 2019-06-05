import cv2
import glob
import os

vedio_path = r'/home/pinlan/wulian/3.mp4'

cap = cv2.VideoCapture(vedio_path)
print(cap.read())

if cap.isOpened():
    success = True
    print('video open successfully')
else:
    success = False
    print('video open failed')

frame_index = 0
interval = 3
img_list = []

# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter(vedio_path[:-4] + '_test_1.avi', fourcc, fps,
#                               (1920, 1080))  # 最后一个是保存图片的尺寸

while success:
    success, frame = cap.read()
    if success and frame_index % interval == 0:
        # im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # im_gamma = cv2.convertScaleAbs(np.power(frame / 255, 0.5) * 255)
        # frame = cv2.flip(frame, -1)
        img_list.append(frame)

    frame_index += 1

cap.release()

if not os.path.exists('/home/pinlan/wulian/image_save/3'):
    os.makedirs('/home/pinlan/wulian/image_save/3')

for i, img in enumerate(img_list):
    # cv2.flip(img, -1)
    cv2.imwrite('/home/pinlan/wulian/image_save/3/3{:05d}.jpg'.format(i), img)

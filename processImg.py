from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图模块
import torch
import os
import pandas as pd
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device= 'cpu'
print('device:', device)

#####是对苹果进行检测#######
model = YOLO('yolov8n.pt')

# 切换计算设备
model.to(device)
# model.cpu()  # CPU
# model.cuda() # GPU
model.device
model.names
#第一问的图像的最后一个是200
ddl=200
#用x,y,w.h格式输入，来确定感兴趣区域
def detect_color(num,img_num,box_num,num_bbox_count,h,img_path,x1,y1,x2,y2):
    image = cv2.imread(img_path)
    #将图像转化成HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #定义感兴趣区域，由YOLOv8输出
    roi = hsv_image[y1:y2, x1:x2]
    # 计算颜色直方图,如果画色相的直方图,'0'是的色相的索引
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    # 计算平均色调
    # 获取Hue通道
    h_channel = roi[:, :, 0]
    # 计算Hue通道的平均值
    average_h_channel = np.mean(h_channel)
    img_num.append(num)
    box_num.append(num_bbox_count)
    h.append(average_h_channel)
    average_color = np.mean(roi, axis=(0, 1))
    plt.plot(hist)
    plt.show()

def process_img(i,img_num,box_num,h,img_path,na):
    num=i
    num=num+1
    results = model(img_path)
    len(results)
    results[0]

    results[0].names
    results[0].boxes.cls
    num_bbox = 0
    num_bbox = len(results[0].boxes.cls)
    results[0].boxes.conf
    results[0].boxes.xyxy
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    bboxes_xywh = results[0].boxes.xywh.cpu().numpy().astype('uint32')
    img_bgr = cv2.imread(img_path)
    #plt.imshow(img_bgr[:,:,::-1])
    #plt.show()
    bbox_color = (150, 0, 0)             # 框的 BGR 颜色
    bbox_thickness = 1                   # 框的线宽

    # 框类别文字
    bbox_labelstr = {
        'font_size':1,         # 字体大小
        'font_thickness':1,   # 字体粗细
        'offset_x':0,          # X 方向，文字偏移距离，向右为正
        'offset_y':-1,        # Y 方向，文字偏移距离，向下为正
    }
    num_bbox_count = 0
    for idx in range(num_bbox): # 遍历每个框
        
        bbox_xyxy = bboxes_xyxy[idx]
        bbox_label = results[0].names[results[0].boxes.cls[idx].item()]

        if bbox_label=='apple':
            num_bbox_count=num_bbox_count+1
            detect_color(num,img_num,box_num,num_bbox_count,h,img_path,bbox_xyxy[0],bbox_xyxy[1],bbox_xyxy[2],bbox_xyxy[3])

            # 画框
            img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)
            img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

    #print(" {} picture is {}".format(num, num_bbox_count))
    #plt.imshow(img_bgr[:,:,::-1])
    #plt.show()
    out_path='save_data\\'+'out'+'_'+na
    cv2.imwrite(out_path, img_bgr)

h=[]
img_num=[]
box_num=[]
for i in range(ddl):
    num=i+1
    num=str(num)
    a='\''
    f='detect_data\\'
    d='.jpg'
    name=f+num+d
    na=num+d
    process_img(i,img_num,box_num,h,name,na)
data = {
    'img_num': img_num,
    'box_num': box_num,
    'h': h
}
df = pd.DataFrame(data)
df.to_excel('mean_h_of_all_apples.xlsx', index=False)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(img_num, box_num, h, c='r', marker='o')
plt.show()
print('pricture has finished')
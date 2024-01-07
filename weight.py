from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图模块
#%matplotlib inline
import torch
import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

#####是对苹果进行检测#######
model = YOLO('yolov8n.pt')


model.to(device)
model.device
model.names
ddl=200
def compute_area(x1,y1,x2,y2):
    x=(x2-x1)/2
    y=(y2-y1)/2
    r=(x+y)/2
    #计算苹果的二维面积
    s=math.pi * r**2
    #返回苹果的二维面积
    return s

def process_img(i,img_num,box_num,area,img_path,na):
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
    bbox_color = (150, 0, 0)             
    bbox_thickness = 1                   

    bbox_labelstr = {
        'font_size':1,         # 字体大小
        'font_thickness':1,   # 字体粗细
        'offset_x':0,          # X 方向，文字偏移距离，向右为正
        'offset_y':-1,        # Y 方向，文字偏移距离，向下为正
    }
    num_bbox_count = 0
    for idx in range(num_bbox): 
        bbox_xyxy = bboxes_xyxy[idx]
        bbox_label = results[0].names[results[0].boxes.cls[idx].item()]

        if bbox_label=='apple':
            num_bbox_count=num_bbox_count+1
            s=compute_area(bbox_xyxy[0],bbox_xyxy[1],bbox_xyxy[2],bbox_xyxy[3])
            img_num.append(num)
            box_num.append(num_bbox_count) 
            area.append(s)
            img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)
            img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

    #y.append(num_bbox_count)
    #print(" {} picture is {}".format(num, num_bbox_count))
    #plt.imshow(img_bgr[:,:,::-1])
    #plt.show()
    out_path='save_data\\'+'out'+'_'+na
    cv2.imwrite(out_path, img_bgr)
img_num=[]
box_num=[]
area=[]
for i in range(ddl):
    num=i+1
    #test_version
    num=str(num)
    a='\''
    f='detect_data\\'
    d='.jpg'
    name=f+num+d
    na=num+d
    process_img(i,img_num,box_num,area,name,na)
weight=[]
divisor = 9.4788
for value in area:
    weight.append(value / divisor)
data = {
    'img_num': img_num,
    'box_num': box_num,
    'area': area,
    'weight':weight
}


df = pd.DataFrame(data)
df.to_excel('area_of_apple.xlsx', index=False)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(img_num, box_num, area, c='r',marker='o'); 
ax.set_xlabel('img_num')
ax.set_ylabel('box_num')
ax.set_zlabel('area')
plt.title('area_of_apple')
plt.show()
plt.hist(weight, bins=30, edgecolor='black')  # bins表示柱子的数量，edgecolor设置柱子边缘颜色

# 设置坐标轴标签和标题
plt.xlabel('weight')
plt.ylabel('Frequency')
plt.title('Histogram')

# 显示图形
plt.show()
print('pricture has finished')
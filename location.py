from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
import torch
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

#####是对苹果进行检测#######
model = YOLO('yolov8n.pt')

model.to(device)
model.device
model.names
ddl=200
def process_img(i,img_path,na,x,y):
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
    bboxes_xyxy

    img_bgr = cv2.imread(img_path)
    plt.imshow(img_bgr[:,:,::-1])
    plt.show()
    bbox_color = (150, 0, 0)             
    bbox_thickness = 1                  

    # 框类别文字
    bbox_labelstr = {
        'font_size':1,         # 字体大小
        'font_thickness':1,   # 字体粗细
        'offset_x':0,          # X 方向，文字偏移距离，向右为正
        'offset_y':-1,        # Y 方向，文字偏移距离，向下为正
    }
    #对检测出来的apple进行计数
    num_bbox_count = 0
    for idx in range(num_bbox): # 遍历每个框
        
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx] 
        
        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[results[0].boxes.cls[idx].item()]
        #如果label是apple的话，那么画框

        if bbox_label=='apple':
            #计数apple
            num_bbox_count=num_bbox_count+1
            #分别求出每个框的中心点
            x_temp=(bbox_xyxy[0]+bbox_xyxy[2])/2
            y_temp=(bbox_xyxy[1]+bbox_xyxy[3])/2
            #将每个框的中心点添加到x和y中，用于之后的散点图画图
            x.append(x_temp)
            y.append(y_temp)
            # 画框
            img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)
            img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

    #print(" {} picture is {}".format(num, num_bbox_count))
    plt.imshow(img_bgr[:,:,::-1])
    plt.show()
    out_path='save_data\\'+'out'+'_'+na
    cv2.imwrite(out_path, img_bgr)

#用来统计每个apple框的x,y坐标
x=[]
y=[]
for i in range(ddl):
    num=i+1
    num=str(num)
    a='\''
    f='detect_data\\'
    d='.jpg'
    name=f+num+d
    na=num+d
    process_img(i,name,na,x,y)

a=list(range(1, 678))
data = {'num': a,'X': x, 'Y': y, }
df = pd.DataFrame(data)
excel_file_path = 'The location of apples.xlsx'
df.to_excel(excel_file_path, index=False)

#画出所有apple的散点图
plt.scatter(x, y)
# 设置图表标题和轴标签
plt.title('The location of all apple')
plt.xlabel('X')
plt.ylabel('Y')
# 显示图表
plt.show()
print('pricture has finished')
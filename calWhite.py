import linecache
import os 
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
# 灰度界限为1
table = [1]*256
table[0] = 0

# 计算标签图白色占比
def calBW(labelpath):
    img=cv2.imread(labelpath,cv2.IMREAD_GRAYSCALE) #灰度图像
    x,y= img.shape
    # print(img.shape) # 图片大小
    black = 0
    white = 0
    #遍历二值图，为0则black+1，否则white+1
    for i in range(x):
        for j in range(y):
            if img[i,j]==0:
                black+=1
            else:
                white+=1
    rate1 = white/(x*y)
    return round(rate1,2)

# 计算标签中心点
def calCenter(labelpath):
    img=cv2.imread(labelpath)
    groundtruth = img[:, :, 0]
    h, w = groundtruth.shape
    contours, cnt = cv2.findContours(groundtruth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    X_center=[]
    Y_center=[]
    for i in range(len(contours)):
        M = cv2.moments(contours[i])  # 计算第一条轮廓的各阶矩,字典形式
        center_x = int(M["m10"] / (M["m00"] + 1e-10))
        center_y = int(M["m01"] / (M["m00"] + 1e-10))
        X_center.append(center_x / w)
        Y_center.append(center_y / h)
    return X_center,Y_center
    
# 计算所有标签图黑白比例均值
def calbw_train(sou_path):
    posneg = '/GT/'
    white = []
    X_center, Y_center = [], []
    for sou in os.listdir(sou_path):
        # test/train/val
        for video in os.listdir(osp.join(sou_path, sou)):
            path = osp.join(sou_path, sou)
            path = osp.join(path, video) + posneg
            for file in os.listdir(path):
                labelpath = osp.join(path, file)
                white.append(calBW(labelpath))
                print(labelpath,calBW(labelpath))
                x_list, y_list = calCenter(labelpath)
                for x in x_list:
                    X_center.append(x)
                for y in y_list:
                    Y_center.append(y)

    mean = sum(white)/len(white)  # 均值
    s2 = np.var(white) # 方差
    std = np.std(white) # 标准差

    filename = "/home/lhq/mycode/data/labels/ASU-Mayo_Clinic-white.txt"
    Note=open(filename,mode='w')
    Note.write(str(white)+'\n') #\n 换行符
    Note.write(str(X_center)+'\n')
    Note.write(str(Y_center)+'\n')
    Note.write(str('mean:%.2f,s2:%.2f,std:%.2f'%(mean,s2,std)))
    Note.close()

    print('%.2f,%.2f,%.2f'%(mean,s2,std))

# 读取文件指定行
def get_line_context(file_path, line_number):
        return linecache.getline(file_path, line_number).strip()

# 获取散点
def getPoint(path):
    # 获取中心点
    x_center = get_line_context(path, 2)
    y_center = get_line_context(path, 3)
    # 转换为list
    x_center = json.loads(x_center)
    y_center = json.loads(y_center)
    return x_center,y_center

# 所有图像相加
def sum_label(sou_path):
    posneg = '/GT/'
    label = []
    for sou in os.listdir(sou_path):
        # test/train/val
        for video in os.listdir(osp.join(sou_path, sou)):
            path = osp.join(sou_path, sou)
            path = osp.join(path, video) + posneg
            for file in os.listdir(path):
                labelpath = osp.join(path, file)
                label.append(cv2.imread(labelpath))
                print(labelpath)
    s_label = sum(label)
    return s_label

if __name__ == '__main__':
    # 计算白色占比、归一化后的标签中心点
    # sou_path = '/data/dataset/lhq/PNSNet/dataset/VPS-TrainSet/ASU-Mayo_Clinic'
    # calbw_train(sou_path)

    # 画散点图
    path = '/home/lhq/mycode/data/labels/'

    # 获取散点
    x_1,y_1 = getPoint(path + 'CVC-ClinicDB-612-white.txt')
    x_2,y_2 = getPoint(path + 'CVC-ColonDB-300-white.txt')
    x_3,y_3 = getPoint(path + 'ASU-Mayo_Clinic-white.txt')
    x_4,y_4 = getPoint(path + 'white.txt')
 
    # 画图
    p1=plt.scatter(x_1,y_1,marker='x',color='g',label='CVC-ClinicDB-612',s=30)
    p2=plt.scatter(x_2,y_2,marker='+',color='r',label='CVC-ColonDB-300',s=30)
    p3=plt.scatter(x_3,y_3,marker='*',color='c',label='ASU-Mayo_Clinic',s=30)
    p3=plt.scatter(x_4,y_4,marker='.',color='b',label='ours',s=30)
    plt.title('mask_center')
    plt.legend(loc = 'upper right')
    # plt.xticks(x_center)
    plt.show()
    plt.savefig(path+"center.png")

    # # 画热力图
    # sou_path = '/data/dataset/lhq/PNSNet/dataset/VPS-TrainSet/ASU-Mayo_Clinic'
    # s_label = sum_label(sou_path)
    # heat_img = cv2.applyColorMap(s_label, cv2.COLORMAP_JET)     #此处的三通道热力图是cv2使用GBR排列
    # cv2.imwrite('/home/lhq/mycode/data/labels/ASU-Mayo_Clinic_heat_img2.png', heat_img)    #cv2保存热力图片

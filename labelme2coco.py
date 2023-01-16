import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import labelme.utils as utils
import numpy as np
import glob
import PIL.Image
import os
import os.path as osp
# import PIL


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./tran.json'):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):

        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num, json_file))
                for shapes in data['shapes']:
                    label = shapes['label']
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    points = shapes['points']  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points.append([points[0][0], points[1][1]])
                    points.append([points[1][0], points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num, json_file):

        image = {}
        # img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        print(json_file.replace("json", "jpg").replace("annotations", "images"))
        imgname = ""
        if os.path.exists(json_file.replace("json", "png").replace("annotations", "images")):
            img = cv2.imread(json_file.replace("json", "png").replace("annotations", "images"), 0)
            imgname = os.path.basename(json_file.replace("json", "png").replace("annotations", "images"))
        else:
            img = cv2.imread(json_file.replace("json", "jpg").replace("annotations", "images"), 0)
            imgname = os.path.basename(json_file.replace("json", "jpg").replace("annotations", "images"))

        # TODO：这里需要指定好输入图像的尺寸，我的图像一般都是同样大小的，所以我就只取一张图片的size
        # img = cv2.imread("/Users/surui/CRT/data/1.jpg", 0)
        height, width = img.shape[:2]

        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = imgname

        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'polyp'
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.getcatid(label)  # 注意，源代码默认为1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示

# json重命名【先把labelme标签放在annotations】
def rename(json_path):
    for type in os.listdir(json_path):
        path = os.path.join(json_path,type)
        for file in os.listdir(path):
            os.rename(os.path.join(path, file), os.path.join(path, 'image'+file[1:]))

# 单个文件转coco格式
def label2coco(json_path):
    for type in os.listdir(json_path):
        path = os.path.join(json_path,type)
        for file in os.listdir(path):
            labelme_json = glob.glob(os.path.join(path, file))
            labelme2coco(labelme_json, os.path.join(path, file))

# arrange
def arrange(sou,tar,data_type):
    # -----创建目标文件目录-----
    image_path = osp.join(tar,data_type)
    label_path = osp.join(tar,'annotations/')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    # -----创建完毕-----
    if 'train' in data_type:
        type = 'train'
        sou_image_path = osp.join(sou,"images","training/")
    elif 'val' in data_type:
        type = 'val'
        sou_image_path = osp.join(sou,"images","validation/")
    else:
        type = 'test'
        sou_image_path = osp.join(sou,"images","test/")
    # 复制图片
    os.system('cp -r ' + sou_image_path+'*' + ' ' + image_path)
    print('image success!')
    # 标签转coco格式, 并合并
    sou_label_path = sou_image_path.replace('images','annotations')
    labelme_json = glob.glob(sou_label_path+'*.json')
    labelme2coco(labelme_json, os.path.join(label_path, 'instance_{}2017.json'.format(type)))
    print(os.path.join(label_path, 'instance_{}2017.json'.format(type))+' success!')


if __name__ == '__main__':
    # json重命名【先把labelme标签放在annotations】
    json_path = "/data/lhq/dataset/polyp_/annotations/"
    rename(json_path)

    # 组织coco文件
    sou = "/data/lhq/dataset/polyp_/"
    tar = "/data/lhq/dataset/coco_/"
    data_type = 'train2017/JPEGImages' # 'val2017/JPEGImages' 'test2017/JPEGImages'
    arrange(sou,tar,data_type)
    

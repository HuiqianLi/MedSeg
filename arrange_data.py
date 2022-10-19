import os 
import os.path as osp
from PIL import Image
height, width = 256, 448
size = (448, 256)
# 灰度界限为1
table = [1]*256
table[0] = 0

# --img
target_list = ['010101','010102','010103','010201','010202','020101','020102','020103','020201',\
    '020301','020302','020303','020304','020401','020402','020403','020404',\
    '020501','020502','020503','020504','020601','020602','020603','020604']
# --video
# test
# target_list = ['030204','030208','020801']
# target_list = ['030204','030207','030208','020801','030303','030308','030309','030312']
# train
# target_list = ['020701','020702','020704','020901','020902','020904','020905','021001',\
#     '030101','030102','030103','040101','040201','040401','040501','040601','040603','040604','040605',\
#     '040701','040801','040901','041001']

def arrange_img():
    # /data/dataset/endoscopy/0101/image1/image47.jpg
    # /data/dataset/endoscopy/0101/image1jsonmask/image47.png

    # 源文件及目标文件
    BASEDIR = "/data/dataset/endoscopy/"
    ENDDIR = "/data/dataset/lhq/PNSNet/dataset/IMG-TrainSet/"

    imnum = 0
    # -----创建目标文件目录-----
    END_img = osp.join(ENDDIR, 'Frame/')
    if not os.path.exists(END_img):
        os.mkdir(END_img)
    END_lab = osp.join(ENDDIR, 'GT/')
    if not os.path.exists(END_lab):
        os.mkdir(END_lab)
    # -----创建完毕-----
    for target in target_list:
        # target[-1] 注意10开头的!!!!!
        try:
            IMGDIR = osp.join(BASEDIR, target[:4] + '/image' + str(int(target[-2:])))
            LABDIR = osp.join(BASEDIR, target[:4] + '/image' + str(int(target[-2:])) + 'jsonmask')
            imgdirs = [name for name in os.listdir(IMGDIR) if name.split('.')[-1] in ['png', 'jpg', 'tif', 'bmp']] # 按扩展名过滤
            for imgdir in imgdirs:
                sour_img = osp.join(IMGDIR, imgdir)  # 找到对应文件目录 
                sour_lab = osp.join(LABDIR, imgdir)  # 找到对应文件目录
                os.system('cp ' + sour_img + ' ' + END_img+'/'+target+'_'+imgdir)
                os.system('cp ' + sour_lab[:-3] + 'png' + ' ' + END_lab+'/'+target+'_'+imgdir[:-3]+'png')
                imnum += 1
                print(imnum)
        except:
            continue
    print('complete ',imnum)

def arrange_test():
    kinds = ['/true/','/false/']

    # 源文件及目标文件
    BASEDIR = "/data/dataset/endoscopy/"
    ENDDIR = "/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/hard/"
    # ENDDIR = "/data/dataset/lhq/PNSNet/dataset/VPS-TrainSet/ours/"

    imnum = 0
    # -----创建目标文件目录-----
    END_img = osp.join(ENDDIR, 'Frame/')
    if not os.path.exists(END_img):
        os.mkdir(END_img)
    END_lab = osp.join(ENDDIR, 'GT/')
    if not os.path.exists(END_lab):
        os.mkdir(END_lab)
    # -----创建完毕-----
    for target in target_list:
        # -----创建目标文件目录-----
        des_img = osp.join(END_img, target)
        if not os.path.exists(des_img):
            os.mkdir(des_img)
        des_lab = osp.join(END_lab, target)
        if not os.path.exists(des_lab):
            os.mkdir(des_lab)
        # -----创建完毕-----
        for kind in kinds:
            # target[-1] 注意10开头的!!!!!
            try:
                IMGDIR = osp.join(BASEDIR, target[:4] + kind + 'image' + str(int(target[-2:])))
                LABDIR = osp.join(BASEDIR, target[:4] + kind + 'image' + str(int(target[-2:])) + 'jsonmask')
                imgdirs = [name for name in os.listdir(IMGDIR) if name.split('.')[-1] in ['png', 'jpg', 'tif', 'bmp']] # 按扩展名过滤
                for imgdir in imgdirs:
                    sour_img = osp.join(IMGDIR, imgdir)  # 找到对应文件目录 
                    sour_lab = osp.join(LABDIR, imgdir)  # 找到对应文件目录 
                    os.system('cp ' + sour_img + ' ' + des_img)
                    os.system('cp ' + sour_lab[:-3] + 'png' + ' ' + des_lab)
                    imnum += 1
                    print(imnum)
            except:
                continue
    print('complete ',imnum)

def arrange_train():
    # kinds = ['/true/','/false/']
    kinds = ['/true/']

    # 源文件及目标文件
    BASEDIR = "/data/dataset/endoscopy/"
    ENDDIR = "/data/dataset/lhq/PNSNet/dataset/VPSW-TrainSet/ours/Train/"

    imnum = 0
    for target in target_list:
        # -----创建目标文件目录-----
        END = osp.join(ENDDIR, target)
        if not os.path.exists(END):
            os.mkdir(END)
        # -----创建完毕-----
        # -----创建目标文件目录-----
        des_img = osp.join(END, 'Frame/')
        if not os.path.exists(des_img):
            os.mkdir(des_img)
        des_lab = osp.join(END, 'GT/')
        if not os.path.exists(des_lab):
            os.mkdir(des_lab)
        # -----创建完毕-----
        for kind in kinds:
            # target[-1] 注意10开头的!!!!!
            try:
                IMGDIR = osp.join(BASEDIR, target[:4] + kind + 'image' + str(int(target[-2:])))
                LABDIR = osp.join(BASEDIR, target[:4] + kind + 'image' + str(int(target[-2:])) + 'jsonmask')
                imgdirs = [name for name in os.listdir(IMGDIR) if name.split('.')[-1] in ['png', 'jpg', 'tif', 'bmp']] # 按扩展名过滤
                for imgdir in imgdirs:
                    sour_img = osp.join(IMGDIR, imgdir)  # 找到对应文件目录 
                    sour_lab = osp.join(LABDIR, imgdir)  # 找到对应文件目录 
                    os.system('cp ' + sour_img + ' ' + des_img)
                    os.system('cp ' + sour_lab[:-3] + 'png' + ' ' + des_lab)
                    imnum += 1
                    print(imnum)
            except:
                continue
    print('complete ',imnum)

# 图片重命名
def rename(sou_path):
    posneg_list = ['/Frame/','/GT/']
    for posneg in posneg_list:
        ENDDIR = sou_path + posneg
        for target in target_list:
            path = osp.join(ENDDIR, target)
            for file in os.listdir(path):
                name = file.split('image')[0]
                num = file.split('image')[1].split('.')[0]
                end = file.split('image')[1].split('.')[1]
                # "image1.jpg", ‘%07d’表示一共7位数
                os.rename(os.path.join(path, file), os.path.join(path, name + 'image' + '%07d' % int(num) + "." + end))
                print(os.path.join(path, name + 'image' + '%07d' % int(num) + "." + end))

# 图片重命名
def rename_train(sou_path):
    posneg_list = ['/Frame/','/GT/']
    for posneg in posneg_list:
        for target in target_list:
            path = osp.join(sou_path, target)
            path = path + posneg
            for file in os.listdir(path):
                name = file.split('image')[0]
                num = file.split('image')[1].split('.')[0]
                end = file.split('image')[1].split('.')[1]
                # "image1.jpg", ‘%07d’表示一共7位数
                os.rename(os.path.join(path, file), os.path.join(path, name + 'image' + '%07d' % int(num) + "." + end))
                print(os.path.join(path, name + 'image' + '%07d' % int(num) + "." + end))

# 缩放并填充
def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例
 
    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)
 
    image = image.resize((nw, nh), Image.BICUBIC)  # 采用双三次插值算法缩小图像
    # new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    new_image = Image.new('RGB', target_size, (0, 0, 0))  # 生成黑色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
 
    return new_image

# 测试图片下采样resize
def resize(sou_path):
    num = 0
    posneg_list = ['/GT/']
    for posneg in posneg_list:
        ENDDIR = sou_path + posneg
        for target in target_list:
            path = osp.join(ENDDIR, target)
            for file in os.listdir(path):
                if posneg == '/Frame/':
                    img = Image.open(osp.join(path, file))
                    img = pad_image(img, size)
                    # img = img.resize((width, height), Image.Resampling.BILINEAR)
                    img.save(osp.join(path, file))
                if posneg == '/GT/':
                    label = Image.open(osp.join(path, file)) 
                    label = pad_image(label, size)
                    label = label.convert('L') # 24位转换为8位
                    # 图片二值化
                    label = label.point(table, '1')
                    # label = label.resize((width, height), Image.NEAREST)
                    label.save(osp.join(path, file))
                num += 1
                print(num)

# 训练图片下采样resize
def resize_train(sou_path):
    num = 0
    posneg_list = ['/Frame/','/GT/']
    for posneg in posneg_list:
        for target in target_list:
            path = osp.join(sou_path, target)
            path = path + posneg
            for file in os.listdir(path):
                if posneg == '/Frame/':
                    img = Image.open(osp.join(path, file))
                    img = pad_image(img, size)
                    # img = img.resize((width, height), Image.Resampling.BILINEAR)
                    img.save(osp.join(path, file))
                if posneg == '/GT/':
                    label = Image.open(osp.join(path, file)) 
                    label = pad_image(label, size)
                    label = label.convert('L') # 24位转换为8位
                    # label = label.resize((width, height), Image.NEAREST)
                    label.save(osp.join(path, file))
                num += 1
                print(num)

# 去掉负样本
def onlyPos(sou_path):
    num = 0
    posneg = '/GT/'
    ENDDIR = sou_path + posneg
    for target in target_list:
        path = osp.join(ENDDIR, target)
        for file in os.listdir(path):
            label = Image.open(osp.join(path, file)) 
            if label.getextrema()==(0, 0):
                # 删掉image和GT
                img_path = sou_path + '/Frame/' + str(target) + '/' + file[:-3] + 'jpg'
                # img_path = sou_path + '/Frame/' + str(target) + '/' + file
                os.remove(osp.join(path, file)) # GT
                os.remove(img_path) # image
            num += 1
            print(num)

# 训练图片转换成8位图
def resize_train(sou_path):
    num = 0
    path = osp.join(sou_path)
    for file in os.listdir(path):
        label = Image.open(osp.join(path, file)) 
        label = label.point(table, '1')
        label = label.convert('L') # 转换为8位
        label.save(osp.join(path, file))
        num += 1
        print(num)

if __name__ == '__main__':
    # arrange_img()
    sou_path = "/data/dataset/lhq/PNSNet/dataset/IMG-TrainSet/GT"
    resize_train(sou_path)
    # img = r"/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/hardrep"
    # onlyPos(img)
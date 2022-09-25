import os 
import os.path as osp
from PIL import Image
height, width = 256, 448
size = (448, 256)

# img
# target_list = ['010101','010102','010103','010201','010202','020101','020102','020103','020201',\
#     '020301','020302','020303','020304','020401','020402','020403','020404',\
#     '020501','020502','020503','020504','020601','020602','020603','020604']
# test
target_list = ['030204','030208','020801']
# target_list = ['030204','030207','030208','020801','030303','030308','030309','030312']
# train
# target_list = ['020701','020702','020704','020901','020902','020904','020905','021001',\
    # '030101','030102','030103','040101','040201','040401','040501','040601','040603','040604','040605',\
    # '040701','040801','040901','041001']

# 缩放并填充
def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例
 
    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)
 
    image = image.resize((nw, nh), Image.BICUBIC)  # 采用双三次插值算法缩小图像
    image.show()
    # new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    new_image = Image.new('RGB', target_size, (0, 0, 0))  # 生成黑色图像
    new_image.show()
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    new_image.show()
 
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
                    # label = label.resize((width, height), Image.NEAREST)
                    label.save(osp.join(path, file))
                num += 1
                print(num)
                
if __name__ == '__main__':
    img = r"/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/hardre"
    resize(img)
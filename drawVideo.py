# 将分割图和原图合在一起
import os
import os.path as osp
from PIL import Image
import numpy as np
import cv2

# mask、预测label画在原图上
def visual(img_path,label_path,pred_path):
    image1 = Image.open(img_path)
    image2 = Image.open(pred_path)
    image3 = Image.open(label_path)

    mat = np.array(image2)
    mat = mat.astype(np.uint8)
    image2 = Image.fromarray(mat, 'P')  
    bin_colormap = [0,0,0] + [255,0,0]*254    # 二值调色板 红色
    image2.putpalette(bin_colormap)

    mat = np.array(image3)
    mat = mat.astype(np.uint8)
    image3 = Image.fromarray(mat, 'P')  
    bin_colormap = [0,0,0] + [255,255,0]*254    # 二值调色板 黄色
    image3.putpalette(bin_colormap)
    
    image2 = image2.convert('RGBA')
    image3 = image3.convert('RGBA') # 转换为4维
    
    #两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    # image = Image.blend(image1,image2,0.3)
    image = Image.blend(image2,image3,0.3)
    image = image.convert('RGB')    # 转换回3维

    # 最后呢我们在创建一个长宽适合两张图片大小的图
    x, y = image.size
    image_out = Image.new('RGB', (x, y*2), (0,0,0))
    image_out.paste(image1,(0,0)) 
    image_out.paste(image,(0,y))

    return np.array(image_out)


if __name__ == "__main__":
    image_base = '/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/hard/Frame/'
    label_base = '/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/hard/GT/'
    pred_base = '/data/dataset/lhq/PNSNet/res/PNS-Net/hard/Pred/'
    out = '/data/dataset/lhq/PNSNet/res/PNS-Net/hard/Pred_draw/'
    if not osp.exists(out):
        os.makedirs(out)
    path_list= os.listdir(image_base)  # 获取该目录下的所有文件名
    # 打开每一个短视频
    for path in path_list:
        image_path = image_base + path + '/'
        label_path = label_base + path + '/'
        pred_path = pred_base + path + '/'

        # 图像、mask、pred目录
        image_index = [name for name in os.listdir(image_path) if name[-3:] in ['png', 'jpg', 'tif', 'bmp']]
        image_index.sort(key=lambda x: int(x[-10:-4]))  # 文件名按数字排序
        # label_index = [name for name in os.listdir(label_path) if name[-3:] == 'png']
        # label_index.sort(key=lambda x: int(x[-7:-4]))  # 文件名按数字排序
        pred_index = [name for name in os.listdir(pred_path) if name[-3:] == 'png']
        pred_index.sort(key=lambda x: int(x[-10:-4]))  # 文件名按数字排序
        test_num = len(pred_index)

        # 新建视频
        file_path = out + path + ".mp4"  # 导出路径
        '''
        fps:
        帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
        如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
        '''
        fps = 25
        size = (448, 256*2)  # 图片的分辨率片
        # size = (1160, 1080)  # 图片的分辨率片
        # size = (384, 288)  # 图片的分辨率片
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
        video = cv2.VideoWriter(file_path, fourcc, fps, size)

        # 将mask画在图像上，并写入每帧
        for index in range(test_num):
            img = image_path + image_index[index]
            label = label_path + pred_index[index]
            pred = pred_path + pred_index[index]
            res = visual(img, label, pred)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)  # 通道顺序为BGR ，注意是BGR
            video.write(res)  # 把图片写进视频
    
        # 保存视频结束
        print(file_path)
        video.release()  # 释放

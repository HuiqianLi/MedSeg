"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
import imgviz
import cv2
import os
import numpy as np
import tqdm


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)
    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_main


def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def Large_Scale_Jittering(mask, img, min_scale=0.1, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w), dtype=np.uint8)
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w]
        return mask_crop, img_crop


def copy_paste(mask_src, img_src, mask_main, img_main, lsj):
    mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # LSJ， Large_Scale_Jittering
    if lsj:
        mask_src, img_src = Large_Scale_Jittering(mask_src, img_src)
        mask_main, img_main = Large_Scale_Jittering(mask_main, img_main)
    else:
        # rescale mask_src/img_src to less than mask_main/img_main's size
        h, w, _ = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)

    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)

    return mask, img


def main():
    input_dir = "../dataset/VOCdevkit2012/VOC2012"
    output_dir = "../dataset/VOCdevkit2012/VOC2012_copy_paste"
    # input path
    segclass = os.path.join(input_dir, 'SegmentationClass')
    JPEGs = os.path.join(input_dir, 'JPEGImages')

    # create output path
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'JPEGImages'), exist_ok=True)

    masks_path = os.listdir(segclass)
    tbar = tqdm.tqdm(masks_path, ncols=100)
    for mask_path in tbar:
        # get source mask and img
        mask_src = np.asarray(Image.open(os.path.join(segclass, mask_path)), dtype=np.uint8)
        img_src = cv2.imread(os.path.join(JPEGs, mask_path.replace('.png', '.jpg')))

        # random choice main mask/img
        mask_main_path = np.random.choice(masks_path)
        mask_main = np.asarray(Image.open(os.path.join(segclass, mask_main_path)), dtype=np.uint8)
        img_main = cv2.imread(os.path.join(JPEGs, mask_main_path.replace('.png', '.jpg')))

        # Copy-Paste data augmentation
        mask, img = copy_paste(mask_src, img_src, mask_main, img_main)

        mask_filename = "copy_paste_" + mask_path
        img_filename = mask_filename.replace('.png', '.jpg')
        save_colored_mask(mask, os.path.join(output_dir, 'SegmentationClass', mask_filename))
        cv2.imwrite(os.path.join(output_dir, 'JPEGImages', img_filename), img)


def mosaic(image_list, mask_list):
    h, w, c = image_list[0].shape  # 获取图像的宽高
    
    '''设置拼接的分隔线位置'''
    min_offset_x = 0.4
    min_offset_y = 0.4  
    scale_low = 1 - min(min_offset_x, min_offset_y)  # 0.6
    scale_high = scale_low + 0.2  # 0.8
 
    image_datas = []  # 存放图像信息
    mask_datas = []  # 存放标注信息
    index = 0  # 当前是第几张图
    
    #（1）图像分割
    for frame,mask in zip(image_list, mask_list):         
        # 对输入图像缩放
        new_ar = w/h  # 图像的宽高比
        scale = np.random.uniform(scale_low, scale_high)   # 缩放0.6--0.8倍
        # 调整后的宽高
        nh = int(scale * h)  # 缩放比例乘以要求的宽高
        nw = int(nh * new_ar)  # 保持原始宽高比例
        # 缩放图像
        frame = cv2.resize(frame, (nw,nh))   
        # 创建一块[416,416]的底版
        new_frame = np.zeros((h,w,3), np.uint8)
        # 确定每张图的位置
        if index==0: new_frame[0:nh, 0:nw, :] = frame   # 第一张位于左上方
        elif index==1: new_frame[0:nh, w-nw:w, :] = frame  # 第二张位于右上方
        elif index==2: new_frame[h-nh:h, 0:nw, :] = frame  # 第三张位于左下方
        elif index==3: new_frame[h-nh:h, w-nw:w, :] = frame  # 第四张位于右下方

        # 缩放图像
        mask = cv2.resize(mask, (nw,nh))   
        # 创建一块[416,416]的底版
        new_mask = np.zeros((h,w), np.uint8)
        # 确定每张图的位置
        if index==0: new_mask[0:nh, 0:nw] = mask   # 第一张位于左上方
        elif index==1: new_mask[0:nh, w-nw:w] = mask  # 第二张位于右上方
        elif index==2: new_mask[h-nh:h, 0:nw] = mask  # 第三张位于左下方
        elif index==3: new_mask[h-nh:h, w-nw:w] = mask  # 第四张位于右下方
    
        index = index + 1  # 处理下一张
        
        # 保存处理后的图像及对应的检测框坐标
        image_datas.append(new_frame)
        mask_datas.append(new_mask)
         
    #（2）将四张图像拼接在一起
    # 在指定范围中选择横纵向分割线
    cutx = np.random.randint(int(w*min_offset_x), int(w*(1-min_offset_x)))
    cuty = np.random.randint(int(h*min_offset_y), int(h*(1-min_offset_y)))        
    
    # 创建一块[416,416]的底版用来组合四张图
    new_image = np.zeros((h,w,3), np.uint8)
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[:cuty, cutx:, :] = image_datas[1][:cuty, cutx:, :]
    new_image[cuty:, :cutx, :] = image_datas[2][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[3][cuty:, cutx:, :]
    # 创建一块[416,416]的底版用来组合四张图
    new_mask = np.zeros((h,w), np.uint8)
    new_mask[:cuty, :cutx] = mask_datas[0][:cuty, :cutx]
    new_mask[:cuty, cutx:] = mask_datas[1][:cuty, cutx:]
    new_mask[cuty:, :cutx] = mask_datas[2][cuty:, :cutx]
    new_mask[cuty:, cutx:] = mask_datas[3][cuty:, cutx:]
    
    return new_mask, new_image
    

if __name__ == '__main__':
    img = '/data/dataset/lhq/PNSNet/dataset/VPS-TrainSet/ours/Train/010101/Frame/image00047.jpg'
    label = '/data/dataset/lhq/PNSNet/dataset/VPS-TrainSet/ours/Train/010101/GT/image00047.png'
    img_ = '/data/dataset/lhq/PNSNet/dataset/VPS-TrainSet/ours/Train/010102/Frame/image00333.jpg'
    label_ = '/data/dataset/lhq/PNSNet/dataset/VPS-TrainSet/ours/Train/010102/GT/image00333.png'
    out = '/data/dataset/lhq/PNSp/test'

    mask_src, img_src, mask_main, img_main =  np.asarray(Image.open(label), dtype=np.uint8), cv2.imread(img), np.asarray(Image.open(label_), dtype=np.uint8), cv2.imread(img_)
    
    # 数据增强-交换颜色
    # img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    # img_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2LAB)
    # mean , std  = img_src.mean(axis=(0,1), keepdims=True), img_src.std(axis=(0,1), keepdims=True)
    # mean2, std2 = img_main.mean(axis=(0,1), keepdims=True), img_main.std(axis=(0,1), keepdims=True)
    # img_src = np.uint8((img_src-mean)/std*std2+mean2)
    # img_main = cv2.cvtColor(img_main, cv2.COLOR_LAB2BGR)
    # img_src = cv2.cvtColor(img_src, cv2.COLOR_LAB2BGR)
    
    # mask, img = copy_paste(mask_src, img_src, mask_main, img_main, lsj=True)
    image_list = [cv2.imread(img)] * 4
    mask_list = [np.asarray(Image.open(label), dtype=np.uint8)] * 4
    mask, img = mosaic(image_list, mask_list)

    cv2.imwrite(out+'/00.jpg', img)
    cv2.imwrite(out+'/00.png', mask)
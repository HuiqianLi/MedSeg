import os
from medpy.metric import binary
import numpy as np
import os.path as osp
from PIL import Image


# dice
def dice(seg, gt):
	return binary.dc(seg, gt)

def hd95(s, g):
    if np.sum(s)==0:
        return 0
    return binary.hd95(s, g, voxelspacing=None)

def compute_ious(s, g):
    '''computes iou for one ground truth mask and predicted mask'''
    intersection = np.logical_and(s, g).sum()
    union = np.logical_or(s, g).sum()
    return np.nanmean(intersection / (union+ 1e-10)) #返回当前图片里所有类的mean iou

if __name__ == '__main__':  
    label_base = '/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/hardre/GT/'
    pred_base = '/data/dataset/lhq/PNSNet/res/PNS-Net/hardre/Pred/'
    # label_base = '/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/CVC-ClinicDB-612-Test/GT/'
    # pred_base = '/data/dataset/lhq/PNSNet/res/PNS-Net/CVC-ClinicDB-612-Test/Pred/'
    out = '/data/dataset/lhq/PNSNet/eval-Result/PNS-Net/'
    if not osp.exists(out):
        os.makedirs(out)
    path_list= os.listdir(label_base)  # 获取该目录下的所有文件夹名
    Note=open(out + '/preds_hardre.txt',mode='w')

    dices,ious,hd95s = [],[],[]
    # 打开每一个短视频
    for path in path_list:
        label_path = label_base + path + '/'
        pred_path = pred_base + path + '/'

        # mask、pred目录
        label_index = [name for name in os.listdir(label_path) if name[-3:] == 'png']
        label_index.sort(key=lambda x: int(x[-10:-4]))  # 文件名按数字排序
        test_num = len(label_index)

        # 计算每一帧的dice、iou
        for index in range(test_num):
            label = label_path + label_index[index]
            pred = pred_path + label_index[index]
            image1 = Image.open(label)
            gt = np.array(image1)
            image2 = Image.open(pred)
            seg = np.array(image2)
            
            if seg.sum()==0:
                continue
            # 求dice
            Dice = dice(seg, gt)
            iou = compute_ious(seg, gt)
            # Hd95 = hd95(seg,gt)
            # print(Hd95)
            if Dice != 0:
                dices.append(Dice)
                ious.append(iou)
            print(label,pred,Dice,iou)

    if len(dices) != 0:
        Note.write("\nmean Dice is " + str("%.4f" %(sum(dices)/len(dices))) + ', ' + "IoU is " + str("%.4f" %(sum(ious)/len(ious))))
        print("\nmean Dice is " + str("%.4f" %(sum(dices)/len(dices))) + ', ' + "IoU is " + str("%.4f" %(sum(ious)/len(ious))))
    Note.close()

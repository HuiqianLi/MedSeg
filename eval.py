import os
from medpy.metric import binary
import numpy as np
import os.path as osp
from PIL import Image


# dice
def dice(seg, gt):
	return binary.dc(seg, gt)

# IOU evaluation
def binary_iou(s, g):
    assert (len(s.shape) == len(g.shape))
    return np.sum(s & g) / (np.sum(s | g) + 1e-10)


if __name__ == '__main__':  

    label_base = '/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/CVC-ClinicDB-612-Test/GT/'
    pred_base = '/data/dataset/lhq/PNSNet/res/PNS-Net/CVC-ClinicDB-612-Test/Pred/'
    out = '/data/dataset/lhq/PNSNet/eval-Result/PNS-Net/'
    if not osp.exists(out):
        os.makedirs(out)
    path_list= os.listdir(label_base)  # 获取该目录下的所有文件夹名
    Note=open(out + '/preds.txt',mode='w')

    meaniou, meandice = [], []
    # 打开每一个短视频
    for path in path_list:
        try:
            label_path = label_base + path + '/'
            pred_path = pred_base + path + '/'

            # mask、pred目录
            label_index = [name for name in os.listdir(label_path) if name[-3:] == 'png']
            label_index.sort(key=lambda x: int(x[-7:-4]))  # 文件名按数字排序
            # pred_index = [name for name in os.listdir(pred_path) if name[-3:] == 'png']
            # pred_index.sort(key=lambda x: int(x[-7:-4]))  # 文件名按数字排序
            test_num = len(label_index)

            dices = []
            ious = []
            # 计算每一帧的dice、iou
            for index in range(test_num):
                label = label_path + label_index[index]
                pred = pred_path + label_index[index]
                image1 = Image.open(label)
                gt = np.array(image1)
                image2 = Image.open(pred)
                seg = np.array(image2)
                
                # 求dice
                Dice = dice(seg, gt)
                iou = binary_iou(seg, gt)
                dices.append(Dice)
                ious.append(iou)
        except:
            continue
        meandice.append(sum(dices)/len(dices))
        meaniou.append(sum(ious)/len(ious))
        Note.write('\n' + path)
        Note.write("\nDice is " + str(meandice[-1]))
        Note.write("\nIoU is " + str(meaniou[-1]))
    Note.write("\nmean Dice is " + str("%.4f" %(sum(meandice)/len(meandice))) + ', ' + "IoU is " + str("%.4f" %(sum(meaniou)/len(meaniou))))
    Note.close()

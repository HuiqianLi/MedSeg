import os
import numpy as np
import os.path as osp
from PIL import Image
from sklearn.metrics import confusion_matrix
import sklearn.metrics
np.seterr(divide='ignore',invalid='ignore')

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric(object):
    def __init__(self):
        self.numClass = 2
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        # self.confusionMatrix = confusion_matrix(gt, seg, labels=[0,1]) # 混淆矩阵
 
    def overallAccuracy(self):
        # return all class overall pixel accuracy,AO评价指标
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
  
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=0) + np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return IoU[0]

    def precision(self):
        # precision = TP / TP + FP
        return sklearn.metrics.precision_score(gt, seg, average='micro')
    
    def recall(self):
        # recall = TP / TP + FN
        return sklearn.metrics.recall_score(gt, seg, average='micro')
    
    def f1(self):
        return sklearn.metrics.f1_score(gt, seg)
    
    # def tps(self):
    #     tn, fp, fn, tp = self.confusionMatrix.ravel()
    #     return tn, fp, fn, tp
    
    def addBatch(self, gt, seg):
        assert gt.shape == seg.shape
        self.confusionMatrix += confusion_matrix(gt, seg, labels=[0,1])
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

if __name__ == '__main__':  
    # label_base = '/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/hardre/GT/'
    # pred_base = '/data/dataset/lhq/PNSNet/res/PNS-Net/hardre/Pred/'
    label_base = '/data/dataset/lhq/PNSNet/dataset/VPS-TestSet/CVC-ClinicDB-612-Test/GT/'
    pred_base = '/data/dataset/lhq/PNSNet/res/PNS-Net/CVC-ClinicDB-612-Test/Pred/'
    # out = '/data/dataset/lhq/PNSNet/eval-Result/PNS-Net/'
    # if not osp.exists(out):
    #     os.makedirs(out)
    path_list= os.listdir(label_base)  # 获取该目录下的所有文件夹名
    # Note=open(out + '/preds.txt',mode='w')

    dices,ious = [],[]
    # 打开每一个短视频
    for path in path_list:
        label_path = label_base + path + '/'
        pred_path = pred_base + path + '/'

        # mask、pred目录
        label_index = [name for name in os.listdir(label_path) if name[-3:] == 'png']
        label_index.sort(key=lambda x: int(x[-10:-4]))  # 文件名按数字排序
        test_num = len(label_index)
        metric = SegmentationMetric()

        # 计算每一帧的dice、iou
        for index in range(test_num):
            label = label_path + label_index[index]
            pred = pred_path + label_index[index]
            gt = np.array(Image.open(label).convert('L')).flatten()
            seg = np.array(Image.open(pred).convert('L')).flatten()

            metric.addBatch(gt, seg)
        oa = metric.overallAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        p = metric.precision()
        mp = np.nanmean(p)
        r = metric.recall()
        mr = np.nanmean(r)
        f1 = (2*p*r) / (p + r)
        mf1 = np.nanmean(f1)

        print(f'oa:{oa}, mIou:{mIoU}, mf1:{mf1}')
        if mf1 != 0 and mIoU != 0:
            dices.append(mf1)
            ious.append(mIoU)

    if len(dices) != 0:
        # Note.write("\nmean Dice is " + str("%.4f" %(sum(dices)/len(dices))) + ', ' + "IoU is " + str("%.4f" %(sum(ious)/len(ious))))
        print("\nmean Dice is " + str("%.4f" %(sum(dices)/len(dices))) + ', ' + "IoU is " + str("%.4f" %(sum(ious)/len(ious))))
    # Note.close()

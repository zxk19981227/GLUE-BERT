import torch
import math

from numpy import mean


class Eval_acc:
    """
    acc的衡量
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, pre: torch.Tensor, real):
        pre=torch.argmax(pre,-1)
        self.correct += (pre == real).cpu().sum()
        self.total += pre.shape[0]

    def result(self):
        """
            返回结果
        """
        print("accuracy is:%s\n" % str(float(self.correct) / float(self.total)))

    def reset(self):
        """
        :return:null
        set all elements to zero
        """
        self.correct = 0
        self.total = 0


class Eval_F1:
    """
        计算F1的函数
    """

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0

    def __call__(self, pre: torch.Tensor, label: torch.Tensor):
        pre=torch.argmax(pre,-1)
        self.TP += torch.sum((pre == 1) & (label == 1)).cpu().item()
        self.TN += torch.sum((pre == 0) & (label == 0)).cpu().item()
        self.FN += torch.sum((pre == 0) & (label == 1)).cpu().item()
        self.FP += torch.sum((pre == 1) & (label == 0)).cpu().item()

    def result(self):
        """
            返回结果
        """
        p = self.TP / (self.TP + self.FP)
        r = self.TP / (self.TP + self.FN)
        f1 = 2 * p * r / (p + r)
        print("f1 score:%s\n" % str(f1))
        print("precision :%s\n" % str(p))
        print("recall:%s\n" % str(r))


    def reset(self):
        """
        :return:null
        set all elements to zero
        """
        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0


class Eval_MMS:
    """
        计算F1的函数
    """

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0

    def __call__(self, pre: torch.Tensor, label: torch.Tensor):
        pre=torch.argmax(pre,-1)
        self.TP += torch.sum((pre == 1) & (label == 1)).cpu().item()
        self.TN += torch.sum((pre == 0) & (label == 0)).cpu().item()
        self.FN += torch.sum((pre == 0) & (label == 1)).cpu().item()
        self.FP += torch.sum((pre == 1) & (label == 0)).cpu().item()


    def result(self):
        up = self.TP * self.TN - self.FP * self.FN
        down = math.sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))
        if down == 0:
            return 0.0
        else:
            return up / down

    def reset(self):
        """
        :return:null
        set all elements to zero
        """
        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0



class Eval_MC:
    def __init__(self):
        self.x = []
        self.y = []

    def __call__(self, pre: torch.Tensor, label: torch.Tensor):
        for x, y in zip(pre, label):
            self.x.append(x.item())
            self.y.append(y.item())

    def result(self):
        """
        返回斯皮尔曼相关系数
        :return:
        """
        mean_x=mean(self.x)
        mean_y=mean(self.y)
        up=0
        down_x=0
        down_y=0
        for x,y in zip(self.x,self.y):
            up+=(x-mean_x)*(y-mean_y)
            down_x+=(x-mean_x)*(x-mean_x)
            down_y+=(y-mean_y)*(y-mean_y)
        if down_x==0 or down_y==0:
            print("down is zero,Error")
            exit(1)
        result=up/(math.sqrt(down_x*down_y))
        print("Spearman Score:%s",str(result))
    def reset(self):
        self.x=[]
        self.y=[]

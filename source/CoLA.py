import torch
import transformers
from CoLA.CoLA_reader import CoLA_reader
from CoLA.CoLA_model import CoLA
from absl import app, flags
import os
import math
import tqdm

from numpy import mean


def compute_score(TP,TN,FN,FP):
    up=TP*TN-FP*FN
    down=math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if down==0:
        return 0.0
    else:
        return up/down

def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # torch.cuda.set_device()
    train_data = CoLA_reader("../glue_data/CoLA/train.tsv",50)
    test_data=CoLA_reader("../glue_data/CoLA/dev.tsv",50)
    bert_name="bert-base-uncased"
    learning_epoch=10
    model=CoLA(bert_name,50,768)
    model=model.cuda()
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),5e-5)
    loader_train=torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
    model.zero_grad()
    optimizer.zero_grad()
    for i in range(learning_epoch):
        print("training epoch ",i)
        loss_sum=[]
        TP=0
        FP=0
        TN=0
        FN=0
        for num,(label,mask,token) in enumerate(loader_train):
            label=label.cuda()
            mask=mask.cuda()
            token=token.cuda()
            pre,loss=model(label,mask,token)
            loss_sum.append(loss.item())
            pre=torch.argmax(pre,-1)
            TP+=torch.sum((pre==1) & (label==1)).cpu().item()
            TN+=torch.sum((pre==0)&(label==0)).cpu().item()
            FN+=torch.sum((pre==0)&(label==1)).cpu().item()
            FP+=torch.sum((pre==1)&(label==0)).cpu().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        M_score=compute_score(TP,TN,FN,FP)
        print("loss:",mean(loss_sum))
        print("\nTP:%s\t,TN:%s\tFN:%s\tFP%s\t\n"%(str(TP),str(TN),str(FN),str(FP)))
        print("M_score_train:%s\n"%str(M_score))
        loader_test=torch.utils.data.DataLoader(test_data,batch_size=32,shuffle=False)
        model.eval()
        loss_sum=[]
        TP=0
        FP=0
        TN=0
        FN=0
        with torch.no_grad():
            print('eval epoch %s\n'%str(i))
            for num, (label, mask, token) in enumerate(loader_test):
                label = label.cuda()
                mask = mask.cuda()
                token = token.cuda()
                pre, loss = model(label, mask, token)
                loss_sum.append(loss.item())
                pre = torch.argmax(pre, -1)
                TP += torch.sum((pre == 1) & (label == 1)).cpu().item()
                TN += torch.sum((pre == 0) & (label == 0)).cpu().item()
                FN += torch.sum((pre == 0) & (label == 1)).cpu().item()
                FP += torch.sum((pre == 1) & (label == 0)).cpu().item()
            print("loss:", mean(loss_sum))
            print("TP:%s\t,TN:%s\tFN:%s\tFP%s\t\n"%(str(TP), str(TN), str(FN), str(FP)))
            M_score = compute_score(TP, TN, FN, FP)
            print("M_score_train:%s\n"%str(M_score))



if __name__ == '__main__':
    app.run(main)

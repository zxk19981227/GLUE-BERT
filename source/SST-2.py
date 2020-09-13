import torch
import transformers
from SST.SST import SST
from SST.SST_reader import SST_reader
from absl import app, flags
import os
import math
from tqdm import tqdm

from numpy import mean




def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # torch.cuda.set_device()
    train_data = SST_reader("../glue_data/SST-2/train.tsv",65)
    test_data=SST_reader("../glue_data/SST-2/dev.tsv",65)
    bert_name="bert-base-uncased"
    learning_epoch=3
    model=SST(bert_name,50,768)
    model=model.cuda()
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),2e-5)
    loader_train=torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
    model.zero_grad()
    optimizer.zero_grad()
    for i in range(learning_epoch):
        print("training epoch ",i)
        loss_sum=[]
        correct=0
        total=0
        for num,(label,mask,token) in tqdm(enumerate(loader_train)):
            label=label.cuda()
            mask=mask.cuda()
            token=token.cuda()
            pre,loss=model(label,mask,token)
            loss_sum.append(loss.item())
            pre=torch.argmax(pre,-1)
            correct+=(pre==label).sum().cpu().item()
            total+=label.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("loss:",mean(loss_sum))
        print("accuracy:%s"%str(correct/total))
        loader_test=torch.utils.data.DataLoader(test_data,batch_size=32,shuffle=False)
        model.eval()
        loss_sum=[]
        correct=0
        total=0
        with torch.no_grad():
            print('eval epoch %s\n'%str(i))
            for num, (label, mask, token) in enumerate(loader_test):
                label = label.cuda()
                mask = mask.cuda()
                token = token.cuda()
                pre, loss = model(label, mask, token)
                loss_sum.append(loss.item())
                pre = torch.argmax(pre, -1)
                correct += (pre == label).sum().cpu().item()
                total += label.shape[0]
            print("loss:", mean(loss_sum))
            print("accuracy:%s" % str(correct / total))



if __name__ == '__main__':
    app.run(main)

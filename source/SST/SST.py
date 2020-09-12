import torch
from transformers import BertModel
class SST(torch.nn.Module):
    def __init__(self,model_name,seq_len,hidden_num):
        super(SST,self).__init__()
        self.Bert=BertModel.from_pretrained(model_name)
        self.out_layer1=torch.nn.Linear(hidden_num,hidden_num//2)
        self.out_layer2=torch.nn.Linear(hidden_num//2,2)
        # self.linear=torch.nn.Linear(hidden_num,2)
        self.loss_funct=torch.nn.CrossEntropyLoss()
    def forward(self,label,mask,token):
        batch_size=label.shape[0]
        # seq_len=token.shape[1]
        out1=self.Bert(input_ids=token,attention_mask=mask)
        out=out1[1]
        out=self.out_layer1(out.view(batch_size,-1))
        out=torch.nn.functional.relu(out)
        out=self.out_layer2(out)#batch,2
        # out=torch.where(mask==1,out,torch.tensor(self.loss_funct.ignore_index))
        # out=self.linear(out)
        loss=self.loss_funct(out.view(batch_size,-1),label)
        return out,loss
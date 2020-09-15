from transformers import BertModel
import torch
class Model(torch.nn.Module):
    def __init__(self, model_name, hidden_num, label_number, loss_function):
        super(Model,self).__init__()
        self.Bert=BertModel.from_pretrained(model_name)
        self.out_layer1=torch.nn.Linear(hidden_num,label_number)
        # self.out_layer2=torch.nn.Linear(hidden_num//2,label_number)
        # self.linear=torch.nn.Linear(hidden_num,2)
        if loss_function=="crossEntropyLoss":
            self.loss_funct=torch.nn.CrossEntropyLoss()
        else:
            self.loss_funct=torch.nn.MSELoss()
    def forward(self,label,mask,segment,token):
        batch_size=label.shape[0]
        # seq_len=token.shape[1]
        out1=self.Bert(input_ids=token,attention_mask=mask,token_type_ids=segment)
        out=out1[1]
        out=self.out_layer1(out.view(batch_size,-1))
        # out=torch.nn.functional.relu(out)
        # out=self.out_layer2(out)#batch,2
        # out=torch.where(mask==1,out,torch.tensor(self.loss_funct.ignore_index))
        # out=self.linear(out)
        loss=self.loss_funct(out,label.view(batch_size))
        return out,loss
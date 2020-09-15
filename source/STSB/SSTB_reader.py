import torch
from transformers import BertTokenizer


class SSTB_reader(torch.utils.data.Dataset):
    def __init__(self,path):
        self.mask=[]
        self.tokens=[]
        self.token=BertTokenizer.from_pretrained("bert-base-uncased")
        self.token_type=[]
        self.label=[]
        tmp_label=[]
        tmp_mask=[]
        tmp_token_type=[]
        tmp_tokens=[]
        max_len=0
        with open(path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines[1:]:
                info=line.strip().split('\t')
                label=float(info[-1])
                sen1=info[-3]
                sen2=info[-2]
                sen1=self.token.tokenize(sen1)
                sen2=self.token.tokenize(sen2)
                token=["CLS"]
                token=token+sen1
                token.append("[SEP]")
                segment=[0]*len(token)
                for each in sen2:
                    segment.append(1)
                    token.append(each)
                segment.append(1)
                token.append("[SEP]")
                mask=[1]*len(token)
                token=self.token.convert_tokens_to_ids(token)
                if len(token) > max_len:
                    max_len = len(token)
                tmp_label.append(label)
                tmp_mask.append(mask)
                tmp_tokens.append(token)
                tmp_token_type.append(segment)
            for label, mask, type, tok in zip(tmp_label, tmp_mask, tmp_token_type, tmp_tokens):
                while len(tok) < max_len:
                    mask.append(0)
                    tok.append(0)
                    type.append(0)

                self.label.append(label)
                self.tokens.append(tok)
                self.mask.append(mask)
                self.token_type.append(type)
                assert len(tok) == len(mask)
                assert len(mask) == len(type)
        self.token_type=torch.tensor(self.token_type)
        self.tokens=torch.tensor(self.tokens)
        self.mask=torch.tensor(self.mask)
        self.label=torch.tensor(self.label)
    def __getitem__(self, item):
        return self.label[item],self.mask[item],self.token_type[item],self.tokens[item]

    def __len__(self):
        return self.label.shape[0]


reader=SSTB_reader("../../glue_data/STS-B/dev.tsv")

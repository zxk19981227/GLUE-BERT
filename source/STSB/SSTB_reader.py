import torch
from transformers import BertTokenizer


class SSTB_reader(torch.utils.data.Dataset):
    def __init__(self,path,max_len):
        self.mask=[]
        self.tokens=[]
        self.token=BertTokenizer.from_pretrained("bert-base-uncased")
        self.segment=[]
        self.label=[]
        with open(path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines[1:]:
                info=line.strip().split('\t')
                lab=float(info[-1])
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
                while len(token)<max_len:
                    token.append(0)
                    mask.append(0)
                    segment.append(0)

                assert len(segment)==len(token)
                assert  len(mask)==len(token)
                self.segment.append(segment)
                self.mask.append(mask)
                self.tokens.append(self.token.convert_tokens_to_ids(token))
                self.label.append(lab)
        self.segment=torch.tensor(self.segment)
        self.tokens=torch.tensor(self.tokens)
        self.mask=torch.tensor(self.mask)
        self.label=torch.tensor(self.label)
    def __getitem__(self, item):
        return self.label[item],self.mask[item],self.segment[item],self.tokens[item]

    def __len__(self):
        return self.label.shape[0]


# reader=SSTB_reader("../../glue_data/STS-B/dev.tsv",200)

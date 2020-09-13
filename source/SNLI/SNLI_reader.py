import torch
from transformers import BertTokenizer
class SNLI_reader(torch.utils.data.Dataset):
    def __init__(self,path,max_seq_len):
        super(SNLI_reader,self).__init__()
        self.tokens=[]
        self.mask=[]
        self.token_type=[]
        self.label=[]
        label_to_dict={"neutral":0,"entailment":1,"contradiction":2}
        self.tokenize=BertTokenizer.from_pretrained("bert-base-uncased")
        with open(path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines[1:]:
                line=line.split('\t')
                sen1=self.tokenize.tokenize(line[8].strip())
                sen2=self.tokenize.tokenize(line[9].strip())
                label=label_to_dict[line[11].strip()]
                token=["CLS"]
                segment=[0]
                mask=[1]
                for each in sen1:
                    token.append(each)
                    segment.append(0)
                    mask.append(1)
                token.append("[SEP]")
                segment.append(0)
                mask.append(1)
                for each in sen2:
                    token.append(each)
                    segment.append(1)
                    mask.append(1)
                token.append("[SEP]")
                segment.append(1)
                mask.append(1)
                token=self.tokenize.convert_tokens_to_ids(token)
                if len(token)>max_seq_len:
                    token=token[:max_seq_len]
                    mask=mask[:max_seq_len]
                    segment=segment[:max_seq_len]
                while len(token)<max_seq_len:
                    token.append(0)
                    mask.append(0)
                    segment.append(0)
                assert len(token)==len(mask)
                assert len(mask)==len(segment)
                self.label.append(label)
                self.mask.append(mask)
                self.tokens.append(token)
                self.token_type.append(segment)
        self.label=torch.tensor(self.label)
        self.mask=torch.tensor(self.mask)
        self.token_type=torch.tensor(self.token_type)
        self.tokens=torch.tensor(self.tokens)


    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, item):
        return self.label[item],self.mask[item],self.token_type[item],self.tokens[item]
# reader=MNLI_reader("../../glue_data/MNLI/examine.tsv",200)


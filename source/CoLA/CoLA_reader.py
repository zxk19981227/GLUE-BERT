import os
import torch
from transformers import BertTokenizer


class CoLA_reader(torch.utils.data.Dataset):
    """
        这个类是用来读取CoLA的分类数据集
    """
    def __init__(self, path):
        self.sentence = []
        self.label = []
        self.ids = []
        self.mask = []
        self.segmenti=[]
        tmp_label=[]
        tmp_mask=[]
        tmp_tokens=[]
        tmp_token_type=[]
        max_len=0
        self.token = BertTokenizer.from_pretrained("bert-base-uncased")
        with open(path, 'r', encoding='utf-8') as f:
            start_indices=["[CLS]"]
            lines = f.readlines()
            for line in lines:
                info = line.strip().split('\t')
                assert (len(info) == 4)
                self.sentence.append(info[3])
                label=int(info[1])
                token = self.token.tokenize(info[3])
                token = self.token.convert_tokens_to_ids(start_indices+token)
                mask = [1] * len(token)
                if max_len<len(token):
                    max_len=len(token)
                tmp_label.append(label)
                tmp_mask.append(mask)
                tmp_tokens.append(token)
            for label, mask,  tok in zip(tmp_label, tmp_mask, tmp_tokens):
                while len(tok) < max_len:
                    mask.append(0)
                    tok.append(0)
                self.label.append(label)
                self.ids.append(tok)
                self.mask.append(mask)
                self.segmenti.append([0]*max_len)
                assert len(tok) == len(mask)

            self.label = torch.tensor(self.label)
            self.mask = torch.tensor(self.mask)
            self.segmenti = torch.tensor(self.segmenti)
            self.ids = torch.tensor(self.ids)
            print("max_len",max_len)
        assert (len(self.sentence) == len(self.label))

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.label[item], self.mask[item],self.segmenti[item], self.ids[item]


# reader = CoLA_reader("../../glue_data/CoLA/train.tsv")

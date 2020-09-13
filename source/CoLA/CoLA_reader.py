import os
import torch
from transformers import BertTokenizer


class CoLA_reader(torch.utils.data.Dataset):
    """
        这个类是用来读取CoLA的分类数据集
    """
    def __init__(self, path, padding_length):
        self.sentence = []
        self.label = []
        self.ids = []
        self.mask = []
        self.segmenti=[]
        self.token = BertTokenizer.from_pretrained("bert-base-uncased")
        with open(path, 'r', encoding='utf-8') as f:
            start_indices=["[CLS]"]
            lines = f.readlines()
            for line in lines:
                info = line.strip().split('\t')
                assert (len(info) == 4)
                self.sentence.append(info[3])
                self.label.append(int(info[1]))
                tokens = self.token.tokenize(info[3])
                tokens = self.token.convert_tokens_to_ids(start_indices+tokens)
                mask = [1] * len(tokens)
                while len(tokens) < padding_length:
                    tokens.append(0)
                    mask.append(0)
                seg=[0] *padding_length
                self.mask.append(mask)
                self.ids.append(tokens)
        self.mask=torch.tensor(self.mask)
        self.ids=torch.tensor(self.ids)
        self.label=torch.tensor(self.label)
        self.segmenti.append(seg)
        self.segmenti=torch.tensor(self.segmenti)
        assert (len(self.sentence) == len(self.label))

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.label[item], self.mask[item],self.segmenti[item], self.ids[item]


# reader = CoLA_reader("../../glue_data/CoLA/examine.tsv", 46)

import os
import torch
from transformers import BertTokenizer


class SST_reader(torch.utils.data.Dataset):
    """
        这个类是用来读取CoLA的分类数据集
    """
    def __init__(self, path, padding_length):
        super(SST_reader,self).__init__()
        self.sentence = []
        self.label = []
        self.ids = []
        self.mask = []
        self.token = BertTokenizer.from_pretrained("bert-base-uncased")
        with open(path, 'r', encoding='utf-8') as f:
            start_indices=["[CLS]"]
            lines = f.readlines()
            for line in lines[1:]:
                info = line.strip().split('\t')
                assert (len(info) == 2)
                self.sentence.append(info[0])
                self.label.append(int(info[1]))
                tokens = self.token.tokenize(info[0])
                tokens = self.token.convert_tokens_to_ids(start_indices+tokens)
                mask = [1] * len(tokens)
                while len(tokens) < padding_length:
                    tokens.append(0)
                    mask.append(0)
                self.mask.append(mask)
                self.ids.append(tokens)
        self.mask=torch.tensor(self.mask)
        self.ids=torch.tensor(self.ids)
        self.label=torch.tensor(self.label)
        assert (len(self.sentence) == len(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.label[item], self.mask[item], self.ids[item]

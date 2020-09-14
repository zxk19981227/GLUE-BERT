import torch
from transformers import BertTokenizer


class MRPC_reader(torch.utils.data.Dataset):
    def __init__(self, path):
        super(MRPC_reader, self).__init__()
        self.tokens = []
        self.mask = []
        self.token_type = []
        self.label = []
        self.tokenize = BertTokenizer.from_pretrained("bert-base-uncased")
        tmp_label=[]
        tmp_mask=[]
        tmp_token_type=[]
        tmp_tokens=[]
        max_len=0
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.split('\t')
                sen1 = self.tokenize.tokenize(line[3].strip())
                sen2 = self.tokenize.tokenize(line[4].strip())
                label = int(line[0].strip())
                token = ["CLS"]
                segment = [0]
                mask = [1]
                for each in sen1:
                    token.append(each)
                    segment.append(0)
                    mask.append(1)
                token.append("[SEP]")
                segment.append(1)
                mask.append(1)
                for each in sen2:
                    token.append(each)
                    segment.append(1)
                    mask.append(1)
                token.append("[SEP]")
                segment.append(1)
                mask.append(1)
                token = self.tokenize.convert_tokens_to_ids(token)
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
        self.label = torch.tensor(self.label)
        self.mask = torch.tensor(self.mask)
        self.token_type = torch.tensor(self.token_type)
        self.tokens = torch.tensor(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.label[item], self.mask[item], self.token_type[item], self.tokens[item]

import torch
from transformers import BertTokenizer


class QNLI_reader(torch.utils.data.Dataset):
    def __init__(self, path, ):
        super(QNLI_reader, self).__init__()
        self.tokens = []
        self.mask = []
        self.token_type = []
        self.label = []
        tmp_label=[]
        tmp_mask=[]
        tmp_token_type=[]
        tmp_tokens=[]
        max_len=0
        self.tokenize = BertTokenizer.from_pretrained("bert-base-uncased")
        with open(path, 'r', encoding='utf-8') as f:
            label_to_dict = {"not_entailment": 0, "entailment": 1}
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split('\t')
                sen1 = self.tokenize.tokenize(line[1].strip())
                sen2 = self.tokenize.tokenize(line[2].strip())
                label = label_to_dict[line[3]]
                token = ["CLS"]
                segment = [0]
                mask = [1]
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
                if label !=1 and label!=0:
                    print(label)
                token = self.tokenize.convert_tokens_to_ids(token)
                if len(token) > max_len:
                    max_len = len(token)
                tmp_label.append(label)
                tmp_mask.append(mask)
                tmp_tokens.append(token)
                tmp_token_type.append(segment)
            if max_len>512:
                max_len=512
            for label, mask, type, tok in zip(tmp_label, tmp_mask, tmp_token_type, tmp_tokens):
                if len(tok)>max_len:
                    continue
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
        print("maxlen:",max_len)
        assert len(self.label)==len(self.mask)
        assert len(self.token_type)==len(self.tokens)
        assert len(self.mask)==len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.label[item], self.mask[item], self.token_type[item], self.tokens[item]
# reader=QNLI_reader("../../glue_data/QNLI/train.tsv")
# loader=torch.utils.data.DataLoader(reader,32,False)
# for each in loader:
#     print(len(each))
#     print("yes")
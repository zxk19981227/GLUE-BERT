from absl import flags,app
from transformers import BertTokenizer
FLAGS=flags.FLAGS
flags.DEFINE_string("file",None,'the file to get')
def main(argv):
    token=BertTokenizer.from_pretrained('bert-base-uncased')
    max_length=0
    # if FLAGS.file:
    with open("../../glue_data/SST-2/train.tsv",'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            info=line.split('\t')
            assert(len(info)==2)
            info=info[0]
            length=len(token.tokenize(info))
            if length>max_length:
                max_length=length
    print(max_length)
if __name__=='__main__':
    app.run(main)
from absl import app, flags
import torch
import os
import math
from tqdm import tqdm
from numpy import mean
from MNLI.MNLI_reader import MNLI_reader
from CoLA.CoLA_reader import CoLA_reader
from SST.SST_reader import SST_reader
from common_model import Model
from evaluate_function import Eval_acc, Eval_F1, Eval_MMS, Eval_MC
from QQP.QQP_reader import QQP_reader
from RTE.RTE_reader import RTE_reader
from STSB.SSTB_reader import SSTB_reader
from SNLI.SNLI_reader import SNLI_reader
from WNLI.WNLI_reader import WNLI_reader
from QNLI.QNLI_reader import QNLI_reader
from MRPC.MRPC_reader import MRPC_reader


FLAGS = flags.FLAGS
flags.DEFINE_string("Task", "QNLI", "the task's name you want to train")
flags.DEFINE_float("learnrate", 5e-5, "the learning recommend to be in 5e-5,4e-5,3e-5,2e-5")
flags.DEFINE_integer("learnepoch", 3, "the learning epoch recommend to use")
flags.DEFINE_string("Bert", "bert-base-uncased", "the bert model decide to use")
flags.DEFINE_integer("GPU",0,"the GPU number use to train")

def train(epoch, rate, model_name, hidden_num, label_number, reader, train_path, eval_path, loss_function,
          accuracy_function):
    """

    :param epoch:
    :param rate:
    :param model_name:
    :param hidden_num:
    :param label_number:
    :param reader:
    :param train_path:
    :param eval_path:
    :param loss_function:
    :param accuracy_function:
    """
    model = Model(model_name, hidden_num, label_number, loss_function)
    # model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),rate)
    train_set = reader(train_path[0])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    acc = accuracy_function()
    acc.reset()
    for i in range(epoch):
        print("process training epoch %s" % str(i))
        loss_it = []
        for num, (label, mask, seg, token) in tqdm(enumerate(train_loader)):
            # label=label.cuda()
            # mask=mask.cuda()
            # seg=seg.cuda()
            # token=token.cuda()
            pre, loss = model(label=label, mask=mask, segment=seg, token=token)
            loss.backward()
            loss_it.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            acc(pre, label)
        print("loss:%s" % str(mean(loss_it)))
        acc.result()
    acc.reset()
    with torch.no_grad():
        for each in eval_path:
            test_set = reader(each)
            loss_it = []
            loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
            print("eval %s" % each)
            for num, (label, mask, seg, token) in tqdm(enumerate(loader)):
                # label=label.cuda()
                # mask=mask.cuda()
                # seg=seg.cuda()
                # token=token.cuda()
                pre, loss = model(label=label, mask=mask, segment=seg, token=token)
                loss_it.append(loss.item())
                acc(pre, label)
            print("loss:%s" % str(mean(loss_it)))
            acc.result()


def main(argvs):
    del argvs
    task_function = {"WNLI": WNLI_reader, "SST-2": SST_reader, "RTE": RTE_reader, "CoLA": CoLA_reader,
                     "MNLI": MNLI_reader, "MRPC": MRPC_reader, "QNLI": QNLI_reader, "QQP": QQP_reader,
                     "STS-B": SSTB_reader,"SNLI":SNLI_reader}
    accuracy_function = {"WNLI": Eval_acc, "SST-2": Eval_acc, "RTE": Eval_acc, "CoLA": Eval_acc, "MNLI": Eval_acc,
                         "MRPC": Eval_F1, "QNLI": Eval_acc, "QQP": Eval_acc, "STS-B": Eval_MC,"SNLI":Eval_acc}
    number_diction = {"SST-2": 2, "RTE": 2, "CoLA": 2, "MNLI": 3, "MRPC": 2, "QNLI": 2, "QQP": 2, "WNLI": 2,"SNLI":3, "STS-B": 1}
    loss_dic = {"SST-2": "crossEntropyLoss", "RTE": "crossEntropyLoss", "CoLA": "crossEntropyLoss",
                "MNLI": "crossEntropyLoss", "MRPC": "crossEntropyLoss", "QNLI": "crossEntropyLoss",
                "QQP": "crossEntropyLoss", "WNLI": "crossEntropyLoss", "STS-B": "MSE","SNLI":"crossEntropyLoss"}
    reader_function = None
    accuracy = None
    number = None
    loss_func = None
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU)
    if FLAGS.Task:
        if FLAGS.Task not in task_function.keys():
            print("Task name not in the glue task name%s" % str(task_function.keys()))
            return
        else:
            reader_function = task_function.get(FLAGS.Task)
            accuracy = accuracy_function.get(FLAGS.Task)
            number = number_diction.get(FLAGS.Task)
            loss_func = loss_dic.get(FLAGS.Task)
    else:
        print("must add Task name")
    train_path = "../glue_data/" + str(FLAGS.Task) + "/train.tsv"
    test_path = "../glue_data/" + str(FLAGS.Task) + "/dev.tsv"
    train_path = [train_path]
    test_path = [test_path]
    if FLAGS.Task == "MNLI":
        test_path = ["../glue_data/MNLI/dev_matched.tsv", "../glue_data/MNLI/dev_mismatched.tsv"]
    train_epoch = FLAGS.learnepoch
    train_rate = FLAGS.learnrate
    train(train_epoch, train_rate, "bert-base-uncased", 768, number, reader_function, train_path, test_path, loss_func,
          accuracy)


if __name__ == "__main__":
    app.run(main)

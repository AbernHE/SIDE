import copy
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

from torch import nn
from torch.autograd import Variable
from torch.utils import data

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(1314)
from argparse import ArgumentParser
from config import BIN_config_SIDE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1")
torch.cuda.set_device(1)

parser = ArgumentParser(description='SIDE Training.')
parser.add_argument('-b', '--batch-size', default=500, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')



def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):    #在这里调用编码   #d[batch_size,50]
        score, loss_con = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        # loss_fct = torch.nn.BCELoss()
        # label_temp = torch.empty(len(label))
        # count = 0
        # for item in label:
        #     label_temp[count] = int(item)
        #     count = count + 1
        # label = Variable(torch.from_numpy(np.array(label_temp)).float()).cuda()
        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()
        loss = torch.nn.functional.binary_cross_entropy(logits, label) #loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    auc = roc_auc_score(y_label, y_pred)
    aupr = average_precision_score(y_label, y_pred)
    # print(y_pred)
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]
    acc = accuracy_score(y_label, y_pred)
    pre = precision_score(y_label, y_pred)
    recall = recall_score(y_label, y_pred)
    f1 = f1_score(y_label, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    if sens == 1 or spec == 0:
        mcc = 0  # 或者设置为其他合适的值
    else:
        mcc = matthews_corrcoef(y_label, y_pred) #mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return auc, aupr, f1, y_pred, loss.item(),acc,pre,recall,sens,spec,mcc


def main():
    config = BIN_config_SIDE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size
    dataset_n = ['mouse','human','fly','mouse_isoform', 'human_gene', 'human_isoform', 'human_gene_1', 'human_gene_2', 'mouse_isoform_l50', 'human_isoform_l50']
                #   0       1      2          3             4             5                 6               7             8                      9

    loss_history = []
    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.workers,
            'drop_last': True}
    avg_acc = []
    avg_auc = []
    avg_aupr = []
    avg_pre = []
    avg_f1 = []
    avg_recall = []
    avg_sens = []
    avg_spec = []
    avg_mcc = []
    index_d = 0
    crossfold = 0
    # dataFolder = get_task(args.task)
    # df_train = pd.read_csv('/home/lab/hcx/SIDE/pro_rna_dataset/human/train.csv')
    # df_val = pd.read_csv('/home/lab/hcx/SIDE/pro_rna_dataset/human/val.csv')
    # df_test = pd.read_csv('/home/lab/hcx/SIDE/pro_rna_dataset/human/test1.csv')
    for crossfold in range(7):
        df_train = pd.read_csv('../SIDE/pro_rna_dataset/'+ dataset_n[index_d] +'_cv1/crossfold_' + str(crossfold) + '/train.csv')
        df_val = pd.read_csv('../SIDE/pro_rna_dataset/'+ dataset_n[index_d] +'_cv1/crossfold_' + str(crossfold) + '/val.csv')
        df_test = pd.read_csv('../SIDE/pro_rna_dataset/'+ dataset_n[index_d] +'_cv1/crossfold_' + str(crossfold) + '/test.csv')
        print('Dataset: '+ dataset_n[index_d] + '_cv1')
        print('crossfold'+str(crossfold) + '-----------------------------------------------------------')
        model = BIN_Interaction_Flat(**config)
        model = model.cuda()

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, dim=0)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        

        training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
        training_generator = data.DataLoader(training_set, **params)

        validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
        validation_generator = data.DataLoader(validation_set, **params)

        testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
        testing_generator = data.DataLoader(testing_set, **params)

        # early stopping
        max_auc = 0
        model_max = copy.deepcopy(model)

        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss,acc,pre,recall,sens,spec,mcc = test(testing_generator, model_max)
            print('Initial Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
                f1) + ' , Test loss: ' + str(loss) + 'Test acc: ' + str(acc) + 'Test Precion: ' + str(pre) + 'Test recall: '+ str(recall)
                + 'Sensitivity: '+ str(sens)+ 'Specificity: '+ str(spec)+ 'MCC: '+ str(mcc))

        print('--- Go for Training ---')
        torch.backends.cudnn.benchmark = True
        for epo in range(args.epochs):
            model.train()
            for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):    #d[16,50]   p[16,545]   d_mask[]
                score, loss_con = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

                label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

                # loss_fct = torch.nn.functional.binary_cross_entropy() # loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score))

                loss = torch.nn.functional.binary_cross_entropy(n, label)
                loss = loss + 0.5*loss_con
                loss_history.append(loss)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if (i % 1000 == 0):
                    print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                        loss.cpu().detach().numpy()))

            # every epoch test
            with torch.set_grad_enabled(False):
                auc, auprc, f1, logits, loss,acc,pre,recall,sens,spec,mcc = test(training_generator, model)
                print('Training at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                    auprc) + ' , F1: ' + str(f1) + ' , Test acc: ' + str(acc) + ' , Test Precion: ' + str(pre) + ' , Test recall: '+ str(recall)
                    + ' , Sensitivity: '+ str(sens)+ ' , Specificity: '+ str(spec)+ ' , MCC: '+ str(mcc))
                auc, auprc, f1, logits, loss,acc,pre,recall,sens,spec,mcc = test(validation_generator, model)
                if auc > max_auc:
                    model_max = copy.deepcopy(model)
                    max_auc = auc
                print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                    auprc) + ' , F1: ' + str(f1) + ' , Test acc: ' + str(acc) + ' , Test Precion: ' + str(pre) + ' , Test recall: '+ str(recall)
                    + ' , Sensitivity: '+ str(sens)+ ' , Specificity: '+ str(spec)+ ' , MCC: '+ str(mcc))
                auc, auprc, f1, logits, loss,acc,pre,recall,sens,spec,mcc = test(testing_generator, model)
                print(
                    'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                        loss) + ' , Test acc: ' + str(acc) + ' , Test Precion: ' + str(pre) + ' , Test recall: '+ str(recall)
                        + ' , Sensitivity: '+ str(sens)+ ' , Specificity: '+ str(spec)+ ' , MCC: '+ str(mcc))

        print('--- Go for Testing ---')
        try:
            with torch.set_grad_enabled(False):
                auc, auprc, f1, logits, loss,acc,pre,recall,sens,spec,mcc = test(testing_generator, model_max)
                print(
                    'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                        loss) + ' , Test acc: ' + str(acc) + ' , Test Precion: ' + str(pre) + ' , Test recall: '+ str(recall)
                        + ' , Sensitivity: '+ str(sens)+ ' , Specificity: '+ str(spec)+ ' , MCC: '+ str(mcc))
                
                avg_auc.append(auc)
                avg_aupr.append(auprc)
                avg_acc.append(acc)
                avg_pre.append(pre)
                avg_f1.append(f1)
                avg_recall.append(recall)
                avg_sens.append(sens)
                avg_spec.append(spec)
                avg_mcc.append(mcc)
        except:
            print('testing failed')
    print("avg_auc",np.sum(avg_auc,axis=0)/len(avg_auc))
    print("avg_auprc",np.sum(avg_aupr,axis=0)/len(avg_aupr))
    print("avg_acc",np.sum(avg_acc,axis=0)/len(avg_auc))
    print("avg_pre",np.sum(avg_pre,axis=0)/len(avg_auc))
    print("avg_recall",np.sum(avg_recall,axis=0)/len(avg_auc))
    print("avg_f1",np.sum(avg_f1,axis=0)/len(avg_auc))
    print("avg_sens",np.sum(avg_sens,axis=0)/len(avg_auc))
    print("avg_spec",np.sum(avg_spec,axis=0)/len(avg_auc))
    print("avg_mcc",np.sum(avg_mcc,axis=0)/len(avg_auc))
    return model_max, loss_history


s = time()
model_max, loss_history = main()
e = time()
print(e - s)

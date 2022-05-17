from class_model import covid_prediction_model
import torch
import esm
import numpy as np
import pandas as pd
from class_dataset import DMS_data,DMS_dataset_pl
from sklearn.model_selection import KFold
from utils import seqs_to_binary_onehot,compute_pppl
import os
from scipy.special import softmax
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,matthews_corrcoef,auc,roc_curve
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import argparse
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


esm_path="/home/chenn0a/chenn0a/covid_esm1b/esm/pretrained_model/esm1b_t33_650M_UR50S.pt"
esm_model,alphabet = esm.pretrained.load_model_and_alphabet_local(esm_path)
#esm_model.cuda()
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('fold', type=int, default=0,help='fold id')
args = parser.parse_args()
out_dir="/home/chenn0a/chenn0a/covid_esm1b/esm_GCN_class/21_5fold4/fold_"+str(args.fold)
training_args = TrainingArguments(
        output_dir=out_dir,   
        num_train_epochs=150,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,   
        warmup_steps=80,                
        weight_decay=0,               
        logging_dir=out_dir+"/logs",           
        logging_steps=10,
        evaluation_strategy="steps",
        report_to = "all",
        save_strategy="steps",
        load_best_model_at_end=True,
        gradient_accumulation_steps=10,
        save_steps=10,
        eval_steps=10,
        metric_for_best_model="eval_f1_ave",
        greater_is_better=True,
	    learning_rate=1e-5,
        save_total_limit=3,
	    eval_accumulation_steps=10,
        #lr_scheduler_type="cosine"
        )
# training_args = TrainingArguments(
#         output_dir=out_dir,   
#         num_train_epochs=100,              
#         per_device_train_batch_size=16,  
#         per_device_eval_batch_size=16,   
#         warmup_steps=50,                
#         weight_decay=0,               
#         logging_dir=out_dir+"/logs",           
#         logging_steps=3,
#         evaluation_strategy="steps",
#         report_to = "all",
#         save_strategy="steps",
#         load_best_model_at_end=True,
#         gradient_accumulation_steps=30,
#         save_steps=3,
#         eval_steps=3,
#         metric_for_best_model="eval_f1_ave",
#         greater_is_better=True,
# 	    learning_rate=1e-5,
#         save_total_limit=3,
# 	    eval_accumulation_steps=30,
#         #lr_scheduler_type="cosine"
#         )


def compute_metrics(eval_pred):
    output,label =eval_pred
    logits,embedding=output
    #y_prob=softmax(logits,axis=2)[:,:,1]
    y_pred_class = np.argmax(logits,axis=2)
    label = np.argmax(label,axis=-1)
    #test_pred=y_pred_class[-6:,:]
    #y_pred_class=y_pred_class[:-6,:]
    #label=label[:-6,:]
    m_dict={}
    for i in range(y_pred_class.shape[1]):
        m_dict['f1'+str(i)]=f1_score(y_true=label[:,i], y_pred=y_pred_class[:,i],average='macro')
        m_dict['precision'+str(i)]=precision_score(y_true=label[:,i], y_pred=y_pred_class[:,i],average='macro')
        m_dict['recall'+str(i)]=recall_score(y_true=label[:,i], y_pred=y_pred_class[:,i],average='macro')
        m_dict['accuracy'+str(i)]=accuracy_score(y_true=label[:,i], y_pred=y_pred_class[:,i])
        m_dict['mcc'+str(i)]=matthews_corrcoef(y_true=label[:,i], y_pred=y_pred_class[:,i])
    # with open(os.path.join(out_dir,'predict_test.txt'),'a+') as f:
    #     f.write(str(test_pred)+'\n')
    m_dict['f1_ave']=np.mean([m_dict['f10'],m_dict['f11'],m_dict['f12'],m_dict['f13'],m_dict['f14'],m_dict['f15'],m_dict['f16'],m_dict['f17'],m_dict['f18']])
    return m_dict

data=DMS_data(data_path=os.path.join(dir,"data/GMM_covid_info_seq.csv"),
                  msa=False)
wt_seq=data.seq
class_label = data.class_label
data=data.get_data()
kf=KFold(n_splits=5,random_state=3,shuffle=True)
# pll=pd.read_csv('/home/chenn0a/chenn0a/covid_esm1b/compute_pppl/pll.csv',index_col=0)
# pll=[[p] for p in pll["pll"]]
# voc_pll=pd.read_csv('/home/chenn0a/chenn0a/covid_esm1b/compute_pppl/covid_pll.csv',index_col=0)
# voc_pll=[[p] for p in voc_pll["pll"]]
f=0
for train_index,test_index in kf.split(data):
    if f != args.fold :
        f=f+1
        continue
    print("Start training fold",f)
    train_seq,val_seq= [data[i] for i in train_index],[data[j] for j in test_index]
    train_labels,val_labels = [class_label[i] for i in train_index],[class_label[j] for j in test_index]
    covid_voc=pd.read_csv(os.path.join(dir,'data/sars-cov-2_variants.csv'),index_col=0)
    covid=[("1",seq) for seq in covid_voc["seq"].to_list()]
    train_idx, train_strs, train_tokens = batch_converter(train_seq)
    val_idx, val_strs, val_tokens = batch_converter(val_seq)
    
    train_dataset=DMS_dataset_pl(train_tokens,train_labels)
    val_dataset=DMS_dataset_pl(val_tokens,val_labels)
    model=covid_prediction_model(esm_path=esm_path,freeze_bert=False)
    trainer = Trainer(
        model=model,                    # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("Compelete fold",f)
    f=f+1
    break


    
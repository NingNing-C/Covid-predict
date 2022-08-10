from model import covid_prediction_model
import torch
import esm
import numpy as np
import pandas as pd
from dataset import DMS_data,DMS_dataset_pl
from sklearn.model_selection import KFold
import os
from scipy.special import softmax
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,matthews_corrcoef
from transformers import Trainer, TrainingArguments
import argparse
from utils.config import *
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


esm_model,alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('fold', type=int, default=0,help='fold id')
parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
args = parser.parse_args()
config = get_config(args)
# out_dir=os.path.join(dir,"result/fold_"+str(args.fold))
out_dir=os.path.join(dir,config.out_dir)
training_args = TrainingArguments(
        output_dir=out_dir,   
        num_train_epochs=config.epochs,              
        per_device_train_batch_size=config.bstch_size,  
        per_device_eval_batch_size=config.bstch_size,   
        warmup_steps=config.warmup_steps,                
        weight_decay=config.weight_decay,               
        logging_dir=out_dir+"/logs",           
        logging_steps=10,
        evaluation_strategy="steps",
        report_to = "all",
        save_strategy="steps",
        load_best_model_at_end=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        metric_for_best_model="eval_f1_ave",
        greater_is_better=True,
	    learning_rate=config.learning_rate,
        save_total_limit=3,
	    eval_accumulation_steps=config.eval_accumulation_steps,
        )


def compute_metrics(eval_pred):
    output,label =eval_pred
    logits,embedding=output
    y_pred_class = np.argmax(logits,axis=2)
    label = np.argmax(label,axis=-1)
    m_dict={}
    for i in range(y_pred_class.shape[1]):
        m_dict['f1'+str(i)]=f1_score(y_true=label[:,i], y_pred=y_pred_class[:,i],average='macro')
        m_dict['precision'+str(i)]=precision_score(y_true=label[:,i], y_pred=y_pred_class[:,i],average='macro')
        m_dict['recall'+str(i)]=recall_score(y_true=label[:,i], y_pred=y_pred_class[:,i],average='macro')
        m_dict['accuracy'+str(i)]=accuracy_score(y_true=label[:,i], y_pred=y_pred_class[:,i])
        m_dict['mcc'+str(i)]=matthews_corrcoef(y_true=label[:,i], y_pred=y_pred_class[:,i])
    m_dict['f1_ave']=np.mean([m_dict['f10'],m_dict['f11'],m_dict['f12'],m_dict['f13'],m_dict['f14'],m_dict['f15'],m_dict['f16'],m_dict['f17'],m_dict['f18']])
    return m_dict

data=DMS_data(data_path=os.path.join(dir,"data/GMM_covid_info_seq.csv"),
                  msa=False)
wt_seq=data.seq
class_label = data.class_label
data=data.get_data()
kf=KFold(n_splits=5,random_state=3,shuffle=True)

f=0
for train_index,test_index in kf.split(data):
    if f != args.fold :
        f=f+1
        continue
    print("Start training fold",f)
    train_seq,val_seq= [data[i] for i in train_index],[data[j] for j in test_index]
    train_labels,val_labels = [class_label[i] for i in train_index],[class_label[j] for j in test_index]
    train_idx, train_strs, train_tokens = batch_converter(train_seq)
    val_idx, val_strs, val_tokens = batch_converter(val_seq)
    
    train_dataset=DMS_dataset_pl(train_tokens,train_labels)
    val_dataset=DMS_dataset_pl(val_tokens,val_labels)
    model=covid_prediction_model(dropout_prob=config.dropout_prob,freeze_bert=False)
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


    
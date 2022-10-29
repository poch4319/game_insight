from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

import json
import pandas as pd 
import gzip 
import altair as alt # for plotting data
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import seaborn as sns
import torch
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split

dict_result = pd.read_table('datav3_dict_pipe.tsv')
dict_result.head()


# cased means it covers both upper and lower cased
# uncased means the case are irrelevant (also the tokenizer within the pipeline lower everything for you so you dont have to lower the data)
cuda_available = torch.cuda.is_available()
model_args = {
    "max_seq_length": 512,
    "evaluate_during_training": True,
    "num_train_epochs": 20,
    "train_batch_size": 20,
    "eval_batch_size": 20,
    "output_dir": "./model/",
    "best_model_dir": "./model/",
    "dataloader_num_workers": 0,
    "use_multiprocessing": False,
    "logging_steps": 10
}
my_model = ClassificationModel("distilbert", "distilbert-base-uncased",  use_cuda=cuda_available, args=model_args, cuda_device=0, num_labels=3)
x = dict_result['review']
y = dict_result['rating_category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
train_df = pd.DataFrame({
    "text": x_train,
    "labels": y_train
})
eval_df = pd.DataFrame({
    "text": x_test,
    "labels": y_test
})

my_model.train_model(train_df, eval_df=eval_df)
import pandas as pd
import numpy as np
import csv 
import string
import nltk 
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

import time
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
from os import path
from sklearn.metrics import hamming_loss, accuracy_score, roc_curve, auc, roc_auc_score, f1_score, multilabel_confusion_matrix, precision_recall_fscore_support

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
from transformers import BertTokenizer
from pathlib import Path
import torch
from torch import Tensor
from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
#import apex
from sklearn.model_selection import train_test_split

import datetime

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc, F1, Hamming_loss
from fast_bert.prediction import BertClassificationPredictor

#Extractor import
import argparse
from OSLextractor.extractor_driver import get_user_cfg
from OSLextractor.repo_extractor import conf, schema
from OSLextractor.repo_extractor.extractor import github_extractor
from OSLextractor.repo_extractor.utils import file_io_utils as file_io

MODEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/output/model_out/'
MINING_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/OSLextractor/docs/example_io/example_cfg.json'

print("Model path local: " + MODEL_PATH)
print("Mining config path local: " + MINING_PATH)

def predictCombinedProjLabels():
    print("Running Bert with all model.")
    LABEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/labels/all/'
    print(LABEL_PATH)
    
    predictor = BertClassificationPredictor(
                model_path=MODEL_PATH,
                label_path=LABEL_PATH, # location for labels.csv file
                multi_label=True,
                model_type='bert',
                do_lower_case=False,
                device=None) # set custom torch.device, defaults to cuda if available

    # Single prediction
    single_prediction = predictor.predict("just get me result for this text")
    print(single_prediction)

    # Batch predictions
    texts = [
        "this is the User interface bug",
        "this is the second text"
        ]

    multiple_predictions = predictor.predict_batch(texts)
    return multiple_predictions
        
def extractIssuesAndClassify():
    """Driver function for GitHub Repo Extractor."""
    tab: str = " " * 4

    cfg_dict: dict = get_user_cfg(MINING_PATH)
    cfg_obj = conf.Cfg(cfg_dict, schema.cfg_schema)

    # init extractor object
    print("\nInitializing extractor...")
    gh_ext = github_extractor.Extractor(cfg_obj)
    print(f"{tab}Extractor initialization complete!\n")

    print("Mining repo data...")
    issuesDict = gh_ext.get_repo_issues_data()
    print(f"\n{tab}Issue data complete!\n")
    print(issuesDict)

    print("Extraction complete!\n")

def runBertPredictions(proj_name):
    if(proj_name == "all"):
        predictCombinedProjLabels()




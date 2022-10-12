from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from os import path
from sklearn.metrics import hamming_loss, accuracy_score, roc_curve, auc, roc_auc_score, f1_score, multilabel_confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np
import torch

#import apex
from fast_bert.prediction import BertClassificationPredictor

#Extractor import
from OSLextractor.extractor_driver import get_user_cfg
from OSLextractor.repo_extractor import conf, schema
from OSLextractor.repo_extractor.extractor import github_extractor
from OSLextractor.repo_extractor.utils import file_io_utils as file_io

MODEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/output/model_out/'
MINING_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/OSLextractor/docs/example_io/example_cfg.json'

print("Model path local: " + MODEL_PATH)
print("Mining config path local: " + MINING_PATH)

def filterLabels(issueLabels):
    labelStr = ""
    i = 0
    for label in issueLabels:
        if label[1] >= .6:
            labelStr += label[0]
            if i != len(issueLabels) - 1 and issueLabels[i + 1][1] >= .6:
                labelStr += ','
        i += 1
    return labelStr
def predictCombinedProjLabels(texts):
    print("Running Bert with all model for test endpoint.")
    LABEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/labels/all/'
    print(LABEL_PATH)
    
    predictor = BertClassificationPredictor(
                model_path=MODEL_PATH,
                label_path=LABEL_PATH, # location for labels.csv file
                multi_label=True,
                model_type='bert',
                do_lower_case=False,
                device=None) # set custom torch.device, defaults to cuda if available

    multiple_predictions = predictor.predict_batch(texts)
    return multiple_predictions
    
def buildIssueArrays(issuesDict):
    issueNums = list(issuesDict.keys())
    issueTexts = []
    index = 0
    for issueNum in issueNums:
        issueTexts.append(issuesDict[issueNum]["body"] + issuesDict[issueNum]["title"])
        index += 1

    return issueNums, issueTexts
    

def classifyMinedIssues(issueNumbers, issueTexts):
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
    single_prediction = predictor.predict(issueTexts[0])

    singlePredDict = {"labels" : single_prediction}

    print("PREDICTION FOR ISSUE: " + issueNumbers[0])
    print(singlePredDict)
    return singlePredDict
    #multiple_predictions = predictor.predict_batch(issueTexts)
    
    #return multiple_predictions

def extractIssuesAndClassify(domains):
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

    issueNumbers, issueTexts = buildIssueArrays(issuesDict, domains)
    print("IssueNumber: " + issueNumbers[0] + " IssueText: " + issueTexts[0])


    return classifyMinedIssues(issueNumbers, issueTexts)

def runBertPredictions(proj_name):
    if(proj_name == "all"):
        predictCombinedProjLabels()




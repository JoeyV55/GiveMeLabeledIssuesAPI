from django.core.management.base import BaseCommand, CommandError
from fast_bert.prediction import BertClassificationPredictor
from GiveMeLabeledIssues.BERT.databaseUtils import *

#TF-IDF imports and cleaning
import os
import pandas as pd


#import apex
from fast_bert.prediction import BertClassificationPredictor

#Extractor import
from OSLextractor.extractor_driver import get_user_cfg
from OSLextractor.repo_extractor import conf, schema
from OSLextractor.repo_extractor.extractor import github_extractor
from OSLextractor.repo_extractor.utils import file_io_utils as file_io

#print(type(os.path.abspath(os.curdir)))
MINING_PATH = (os.path.abspath(os.curdir)) + '/OSLextractor/docs/example_io/example_cfg.json'
MODEL_PATH = (os.path.abspath(os.curdir)) + '/GiveMeLabeledIssues/BERT/BERTModels/output/model_out/'
LABEL_PATH = (os.path.abspath(os.curdir)) + '/GiveMeLabeledIssues/BERT/BERTModels/labels/all/'
print("MODEL PATH: " + MODEL_PATH)
print("LABEL PATH: " + LABEL_PATH)
PROBABILITY_THRESHOLD = .5
ISSUE_LIMIT = 10
class Command(BaseCommand):
    help = 'Populates the project issue database tables.'

    def add_arguments(self, parser):
        parser.add_argument('project')
        parser.add_argument('classifier')

    #Used for BERT  
    def buildIssueArrays(self, issuesDict):
        issueNums = list(issuesDict.keys())
        issueTexts = []
        issueTitles = []
        index = 0
        #print("ISSUESDict: ", issuesDict)

        for issueNum in issueNums:
            titleText = "" 
            bodyText = ""  

            if issuesDict[issueNum]["title"] is not None:
                titleText = issuesDict[issueNum]["title"] 
            if issuesDict[issueNum]["body"] is not None:
                bodyText = issuesDict[issueNum]["body"]

            print("BODY TEXT: ", issuesDict[issueNum]["body"])
            print("TITLE TEXT: ", titleText) 
            issueTexts.append(bodyText + titleText)
            issueTitles.append(titleText)
            index += 1

        return issueNums, issueTexts, issueTitles
    
    #Used for TF-IDF
    def buildIssueDf(self, issuesDict):
        issueNums = list(issuesDict.keys())
        issueTexts = []
        issueTitles = []

        for issueNum in issueNums:
            titleText = "" 
            bodyText = ""  
            if issuesDict[issueNum]["title"] is not None:
                titleText = issuesDict[issueNum]["title"] 
            if issuesDict[issueNum]["body"] is not None:
                bodyText = issuesDict[issueNum]["body"] 
            issueTexts.append(bodyText + titleText)
            issueTitles.append(titleText)
            
        issuesDict = {"IssueNumber" : issueNums, "IssueText" : issueTexts}
        issuesDf = pd.DataFrame(data=issuesDict)
        return issuesDf

    def filterLabels(self, issueLabels):
        labelStr = ""
        i = 0
        for label in issueLabels:
        # print(label)
            if label[1] >= PROBABILITY_THRESHOLD:
                labelStr += label[0]
                if i != len(issueLabels) - 1 and issueLabels[i + 1][1] >= PROBABILITY_THRESHOLD:
                    labelStr += ','
            i += 1
        return labelStr

    def classifyMinedIssuesBERT(self, issueNumbers, issueTexts, issueTitles, project):
        print("Running Bert with all model.")

        #print(LABEL_PATH)
        predictor = BertClassificationPredictor(
                    model_path=MODEL_PATH,
                    label_path=LABEL_PATH, # location for labels.csv file
                    multi_label=True,
                    model_type='bert',
                    do_lower_case=False,
                    device=None) # set custom torch.device, defaults to cuda if available

        issues = []
        # print("ISSUETITLES: ", issueTitles)
        # print("ISSUETexts: ", issueTexts)
        # print("ISSUENUMBERS: ", issueNumbers)
        #print(domains)
        requestVals = {"issues": []}
        for i in range(0, len(issueTitles)):
            print("ISSUE CLASSIFYING: ", i)
            
            labelStr = self.filterLabels(predictor.predict(issueTexts[i]))
            issueDict = {}
            issueDict["issueTitle"] = issueTitles[i]
            issueDict["issueNumber"] = issueNumbers[i]
            issueDict["issueText"] = issueTexts[i]
        
            issueDict["issueLabels"] = labelStr
            persistToDB(issueDict, project)

            i += 1
            if i > ISSUE_LIMIT:
                break

        return requestVals        

    def extractIssuesAndClassify(self, project, classifier):
        """Driver function for GitHub Repo Extractor."""
        tab: str = " " * 4

        cfg_dict: dict = get_user_cfg(MINING_PATH)
        cfg_obj = conf.Cfg(cfg_dict, schema.cfg_schema)
        print(os.path.abspath(os.curdir))
        os.chdir("../")
        print(os.path.abspath(os.curdir))
        # init extractor object
        print("\nInitializing extractor...")
        gh_ext = github_extractor.Extractor(project, cfg_obj)
        print(f"{tab}Extractor initialization complete!\n")

        print("Mining repo data...")
        issuesDict = gh_ext.get_repo_issues_data()
        print(f"\n{tab}Issue data complete!\n")

        if classifier == 'BERT':
            issueNumbers, issueTexts, issueTitles = self.buildIssueArrays(issuesDict)
            print("IssueNumber: " + issueNumbers[0] + " IssueText: " + issueTexts[0])
            return self.classifyMinedIssuesBERT(issueNumbers, issueTexts, issueTitles, project)
        
        #Otherwise do TF-IDF, TODO make elif when Doc2Vec is inserted.
        issueDf = self.buildIssueDf(issuesDict)
        #Run TF-IDF classification with issueDf. 
    
    def handle(self, *args, **options):
        valid_projects = {'jabref' : "JabRef/jabref", 'powertoys' : "microsoft/PowerToys", 'audacity' : "audacity/audacity"}
        valid_classifiers = ['BERT', 'TFIDF']

        if options['project'] not in valid_projects:
            print("Invalid project name. Valid projects are:", end=' ')
            for project in valid_projects.keys():
                print(project, end=' ')
            return
        if options['classifier'] not in valid_classifiers:
            print("Invalid project name. Valid projects are:", end=' ')
            for classifier in valid_classifiers:
                print(classifier, end=' ')
            return
        
        self.extractIssuesAndClassify(valid_projects[options['project']], options['classifier'])

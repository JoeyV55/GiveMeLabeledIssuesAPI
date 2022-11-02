from django.core.management.base import BaseCommand, CommandError
from fast_bert.prediction import BertClassificationPredictor

from GiveMeLabeledIssues.BERT.bertModelRunner import *
from GiveMeLabeledIssues.BERT.databaseUtils import *

class Command(BaseCommand):
    help = 'Populates the JabRef issue database.'
    #Testing limit on number of issues classified.
    issueLimit = 10
    threshold = .5

    def add_arguments(self, parser):
        parser.add_argument('project')

    def classifyMinedIssues(issueNumbers, issueTexts, issueTitles, project):
        print("Running Bert with all model.")
        MODEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/output/model_out/'
        LABEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/labels/all/'
        print(LABEL_PATH)
        print("TEST")
        predictor = BertClassificationPredictor(
                    model_path=MODEL_PATH,
                    label_path=LABEL_PATH, # location for labels.csv file
                    multi_label=True,
                    model_type='bert',
                    do_lower_case=False,
                    device=None) # set custom torch.device, defaults to cuda if available

        issues = []
        print("ISSUETITLES: ", issueTitles)
        print("ISSUETexts: ", issueTexts)
        print("ISSUENUMBERS: ", issueNumbers)
        #print(domains)
        requestVals = {"issues": []}
        for i in range(0, len(issueTitles)):
            print("ISSUE CLASSIFYING: ", i)
            
            labelStr = filterLabels(predictor.predict(issueTexts[i]))
            issueDict = {}
            issueDict["issueTitle"] = issueTitles[i]
            issueDict["issueNumber"] = issueNumbers[i]
            issueDict["issueText"] = issueTexts[i]
        
            issueDict["labels"] = labelStr
            persistToDB(issueDict, project)

            i += 1
            if i == issueLimit:
                break
        
        print("RequestVals: " + str(requestVals))

        return requestVals
        #multiple_predictions = predictor.predict_batch(issueTexts)
        
        #return multiple_predictions

    def extractIssuesAndClassify(project):
        """Driver function for GitHub Repo Extractor."""
        
        tab: str = " " * 4

        cfg_dict: dict = get_user_cfg(MINING_PATH)
        cfg_obj = conf.Cfg(cfg_dict, schema.cfg_schema)

        # init extractor object
        print("\nInitializing extractor...")
        gh_ext = github_extractor.Extractor(project, cfg_obj)
        print(f"{tab}Extractor initialization complete!\n")

        print("Mining repo data...")
        issuesDict = gh_ext.get_repo_issues_data()
        print(f"\n{tab}Issue data complete!\n")

        issueNumbers, issueTexts, issueTitles = buildIssueArrays(issuesDict)
        print("IssueNumber: " + issueNumbers[0] + " IssueText: " + issueTexts[0])


        return classifyMinedIssues(issueNumbers, issueTexts, issueTitles, project)

        
    def handle(self, *args, **options):
        print(options['project'])

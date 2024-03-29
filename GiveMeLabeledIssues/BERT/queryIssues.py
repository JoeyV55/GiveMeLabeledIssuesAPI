from GiveMeLabeledIssues.models import *
MODEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/output/model_out/'
MINING_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/OSLextractor/docs/example_io/example_cfg.json'


#Testing limit on number of issues classified.
issueLimit = 10
threshold = .5
    
def findIssues(project, labels):
    print("Running Bert with all model.")
    LABEL_PATH = '/mnt/e/RESEARCH/GRAD/GiveMeLabeledIssuesAPI/GiveMeLabeledIssues/BERT/labels/all/'
    labelsDict = {}
    projectQs = []
    issues = []
    requestVals = {"issues": []}

    if project == "JabRef/jabref":
        projectQs = JabRefIssue.objects.filter(issueLabels__contains=labels[0])

        for label in labels:
            currQs = JabRefIssue.objects.filter(issueLabels__contains=label)
            print("LABEL: ", label, " currQS: ", currQs)
            projectQs = projectQs.union(currQs)
            print("ProjectQs: ", projectQs)

        for issue in projectQs:
            print(issue.issueText)

        
    elif project == "microsoft/PowerToys":
        projectQs = PowerToysIssue.objects.filter(issueLabels__contains=labels[0])

        for label in labels:
            currQs = PowerToysIssue.objects.filter(issueLabels__contains=label)
            projectQs = projectQs.union(currQs)
    
    elif project == "audacity/audacity":
        projectQs = AudacityIssue.objects.filter(issueLabels__contains=labels[0])

        for label in labels:
            currQs = AudacityIssue.objects.filter(issueLabels__contains=label)
            projectQs = projectQs.union(currQs)
    
    for issue in projectQs:
        labelStr = issue.issueLabels
        issueDict = {}
        issueDict["issueTitle"] = issue.issueTitle
        issueDict["issueNumber"] = issue.issueNumber
        issueDict["issueText"] = issue.issueText
        issueDict["issueLabels"] = labelStr
        requestVals["issues"].append(issueDict)
    return requestVals




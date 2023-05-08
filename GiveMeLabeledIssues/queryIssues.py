from GiveMeLabeledIssues.models import *


#Testing limit on number of issues classified.
issueLimit = 10
threshold = .5
    
def findIssues(project, labels):
    print("Finding issues with labels: ", labels)
    labelsDict = {}
    projectQs = {}
    issues = []
    requestVals = {"issues": []}

    if project == "JabRef/jabref":
        projectQs = JabRefIssue.objects.filter(issueLabels__contains=labels[0])
        print("QS LEN for first label ", labels[0], ": ", len(projectQs))

        for i in range(1, len(labels)):
            currQs = JabRefIssue.objects.filter(issueLabels__contains=labels[i])
            print("CURR QS LEN for label ",labels[i], ": ", len(currQs))
            projectQs = projectQs | currQs
            print("Project QS after union with set for label: ", labels[i], ": ", len(projectQs))

        # for issue in projectQs:
        #     print(issue.issueText)

        
    elif project == "microsoft/PowerToys":
        projectQs = PowerToysIssue.objects.filter(issueLabels__contains=labels[0])

        for i in range(1, len(labels)):
            currQs = PowerToysIssue.objects.filter(issueLabels__contains=labels[i])
            projectQs = projectQs | currQs
    
    j = 0
    for issue in projectQs:

        labelStr = issue.issueLabels
        issueDict = {}
        issueDict["issueTitle"] = issue.issueTitle
        issueDict["issueNumber"] = issue.issueNumber
        issueDict["issueText"] = issue.issueText
        issueDict["issueLabels"] = labelStr.rstrip(',')
        requestVals["issues"].append(issueDict)
        print("ISSUE: ", issue.issueNumber, " Labels: ", labelStr)        
        j+=1
    return requestVals




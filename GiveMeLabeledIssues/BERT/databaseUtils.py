from GiveMeLabeledIssues.models import *

def persistToDB(issueNumber, labelStr, project):
    labels = labelStr.split(',')

    if project == "jabref":
        storeJabRefIssue(issueNumber, labels, project)
    
    elif project == "powertoys":
        storePowerToysIssue(issueNumber, labels, project)

def storeJabRefIssue(issueNumber, labels, project):
    try:
        query = JabRefIssue.objects.get(issueNumber = issueNumber)
    except JabRefIssue.DoesNotExist:
        print("Couldn't find issue", issueNumber)

def storePowerToysIssue(issueNumber, labels, project):
    try:
        query = PowerToysIssue.objects.get(issueNumber = issueNumber)
    except PowerToysIssue.DoesNotExist:
        print("Couldn't find issue", issueNumber)

from GiveMeLabeledIssues.models import *

def persistToDB(issueDict, project):

    if project == "JabRef/jabref":
        storeJabRefIssue(issueDict, project)
    
    elif project == "Powertoys/Powertoys":
        storePowerToysIssue(issueDict, project)

def storeJabRefIssue(issueDict, project):
    labels = issueDict["issueLabels"].split(',')
    
    Util = 1 if "Util" in labels else 0
    NLP = 1 if "NLP" in labels else 0
    APM = 1 if "APM" in labels else 0
    Network = 1 if "Network" in labels else 0
    DB = 1 if "DB" in labels else 0
    Interpreter = 1 if "Interpreter" in labels else 0
    Logging = 1 if "Logging" in labels else 0
    Data_Structure = 1 if "Data.Structure" in labels else 0
    i18n = 1 if "i18n" in labels else 0
    DevOps = 1 if "DevOps" in labels else 0
    Logic = 1 if "Logic" in labels else 0
    Microservices = 1 if "Microservices" in labels else 0
    Test = 1 if "Test" in labels else 0
    Search = 1 if "Search" in labels else 0
    IO = 1 if "IO" in labels else 0
    UI = 1 if "UI" in labels else 0
    Parser = 1 if "Parser" in labels else 0
    Security = 1 if "Security" in labels else 0
    App = 1 if "App" in labels else 0

    newIssue = {"issueNumber": issueDict["issueNumber"], "issueTitle" : issueDict["issueTitle"], 
    "issueText" : issueDict["issueText"], "issueLabels": issueDict["issueLabels"], "Util": Util, "NLP" : NLP, "APM" : APM, "Network" : Network, 
    "DB": DB, "Interpreter" : Interpreter, "Logging" : Logging, "Data_Structure" : Data_Structure, "i18n" : i18n,
     "DevOps" : DevOps, "Logic" : Logic, "Microservices" : Microservices, "Test" : Test, "Search": Search,
      "IO": IO, "UI" : UI, "Parser" : Parser, "Security" : Security, "App" : App}

    JabRefIssue.objects.update_or_create(issueNumber=issueDict["issueNumber"], defaults=newIssue)


def storePowerToysIssue(issueDict, project):
    labels = issueDict["issueLabels"].split(',')

    PowerToysIssue.objects.update_or_create()
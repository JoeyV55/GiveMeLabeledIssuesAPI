from django.db import models

# Create your models here.
class JabRefIssue(models.Model):
    #Number
    issueNumber = models.IntegerField(primary_key = True, db_column='IssueNumber')
    #Title
    issueTitle = models.TextField(default='')
    #Text
    issueText = models.TextField(default='')
    #LabelStr
    issueLabels = models.TextField(default='')
    #Labels 
    Util = models.BooleanField()
    NLP = models.BooleanField()
    APM = models.BooleanField()
    Network = models.BooleanField()
    DB = models.BooleanField()
    Interpreter = models.BooleanField()
    Logging = models.BooleanField()
    Data_Structure = models.BooleanField(db_column = 'Data.Structure')
    i18n = models.BooleanField()
    DevOps = models.BooleanField()
    Logic = models.BooleanField()
    Microservices = models.BooleanField()
    Test = models.BooleanField()
    Search = models.BooleanField()
    IO = models.BooleanField()
    UI = models.BooleanField()
    Parser = models.BooleanField()
    Security = models.BooleanField()
    App = models.BooleanField()

    def __str__(self): # __str__ for Python 3, __unicode__ for Python 2
        return str(self.issueNumber)


class PowerToysIssue(models.Model):
    #Number
    issueNumber = models.IntegerField(primary_key = True, db_column='IssueNumber')
    #Title
    issueTitle = models.TextField(default='')
    #Text
    issueText = models.TextField(default='')
    #LabelStr
    issueLabels = models.TextField(default='')
    #Labels 
    APM = models.BooleanField()
    Interpreter = models.BooleanField()
    Logging = models.BooleanField()
    Data_Structure = models.BooleanField(db_column = 'Data.Structure')
    i18n = models.BooleanField()
    Setup = models.BooleanField()
    Logic = models.BooleanField()
    Microservices = models.BooleanField()
    Test = models.BooleanField()
    Search = models.BooleanField()
    UI = models.BooleanField()
    Parser = models.BooleanField()
    App = models.BooleanField()

    def __str__(self): # __str__ for Python 3, __unicode__ for Python 2
        return str(self.issueNumber)

class AudacityIssue(models.Model):
    #Number
    issueNumber = models.IntegerField(primary_key = True, db_column='IssueNumber')
    #Title
    issueTitle = models.TextField(default='')
    #Text
    issueText = models.TextField(default='')
    #LabelStr
    issueLabels = models.TextField(default='')
    #Labels 
    Util = models.BooleanField()
    APM = models.BooleanField()
    Network = models.BooleanField()
    DB = models.BooleanField()
    Error_Handling = models.BooleanField(db_column = 'Error.Handling')
    Logging = models.BooleanField()
    Lang = models.BooleanField()
    Data_Structure = models.BooleanField(db_column = 'Data.Structure')
    i18n = models.BooleanField()
    Setup = models.BooleanField()
    Logic = models.BooleanField()
    IO = models.BooleanField()
    UI = models.BooleanField()
    Parser = models.BooleanField()
    Event_Handling = models.BooleanField(db_column = 'Event.Handling')
    App = models.BooleanField()
    GIS = models.BooleanField()
    Multimedia = models.BooleanField()
    CG = models.BooleanField()

    def __str__(self): # __str__ for Python 3, __unicode__ for Python 2
        return str(self.issueNumber)
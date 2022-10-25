from django.db import models

# Create your models here.
class JabRefIssue(models.Model):
    #Number
    issueNumber = models.IntegerField(primary_key = True, db_column='IssueNumber')

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


class PowerToysIssue(models.Model):
    #Number
    issueNumber = models.IntegerField(primary_key = True, db_column='IssueNumber')

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
from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets, views
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework import status
from GiveMeLabeledIssues.BERT.serializers import UserSerializer, GroupSerializer, BERTRequestSerializer
from GiveMeLabeledIssues.BERT.bertModelRunner import *;
import json


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


class BERTRequestView(views.APIView):
    def get(self, request, project, domains):
        print("Hit BERT endpoint with GET request!")
        #Test titles for issues 9191 and 9192
        titles = [
        "Allow choosing group when import from browser extension",
        "Change dark theme colour of highlighted text in Entry Merge Dialogue"
        ]
        res = predictCombinedProjLabels(titles)
        issueListStr = ""
        i = 0
        issues = []
        print(project)
        print(domains)
        requestVals = {"issues": []}
        for issue in res:
            issueDict = {}
            issueDict["title"] = titles[i]
            issueDict["issueNumber"] = 9191 + i
            labelStr = filterLabels(issue)
            issueDict["labels"] = labelStr
            i += 1
            requestVals["issues"].append(issueDict)
        
        print("ISSUES: " + str(requestVals))
        return Response(requestVals, status = status.HTTP_200_OK)


class MineIssuesView(views.APIView):
    def get(self, request, project, domains):
        print("Hit Mine endpoint with GET request!")
        domainsList = domains.split(',')
        project = project.replace(',', '/')
        print(project)
        print(domains)
        res = extractIssuesAndClassify(project, domainsList)
        #return Response({"Inputted project name as: " + project + " and domains as: " + domains + "\n Output: " + json.dumps(res)}, status = status.HTTP_200_OK)

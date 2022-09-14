from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets, views
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework import status
from GiveMeLabeledIssues.BERT.serializers import UserSerializer, GroupSerializer, BERTRequestSerializer
from GiveMeLabeledIssues.BERT.bertModelRunner import *;


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
        res = predictCombinedProjLabels()
        issueListStr = ""
        i = 1
        for issue in res:
            issueListStr += "Issue " + str(i) + ": " + str(issue) + " \n"
            i += 1
        print("ISSUESTR: " + issueListStr)
        return Response({"Inputted project name as: " + project + " and domains as: " + domains + "\n Output: " + issueListStr}, status = status.HTTP_200_OK)

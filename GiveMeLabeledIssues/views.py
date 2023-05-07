from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets, views
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework import status
from GiveMeLabeledIssues.serializers import UserSerializer, GroupSerializer, BERTRequestSerializer
from GiveMeLabeledIssues.queryIssues import *;
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

class QueryIssuesView(views.APIView):
    def get(self, request, project, domains):
        domainsList = domains.split(',')
        print("Hit QueryIssues endpoint with GET")
        project = project.replace(',', '/')
        print(project)
        print(domains)
        results = findIssues(project, domainsList)
        return Response(results, status = status.HTTP_200_OK)

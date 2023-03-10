# GiveMeLabeledIssuesAPI
This is the backend REST API interface that accepts requests to mine and classify open source issues from supported projects. The REST API allows access to both the BERT and TF-IDF text classification models. 

[![DOI](https://zenodo.org/badge/486040723.svg)](https://zenodo.org/badge/latestdoi/486040723)


# How to install

Run these commands to run to serve and run the API
1. Review the system requirements and install the packages needed to run [fast-bert](https://github.com/utterworks/fast-bert)
2. The packages needed to run the API should be within the issuesenv included in this repo. 
  
  <blockquote>source <PATH TO issuesenv>/bin/activate</blockquote>  
  
3. Run the server:

<blockquote>python manage.py runserver</blockquote>


# Architecture

### Proof of Concept
![alt text](https://github.com/JoeyV55/GiveMeLabeledIssuesAPI/blob/master/GiveMeLabeledIssuesPOC.png "POC Architecture")

GiveMeLabeledIssues is a fully integrated Open Source Issue recommendation system. GiveMeLabeledIssues gives an interface to utilize the trained and tested BERT and TF-IDF machine learning models to label OSS issues. The system starts with a front end web application that prompts users to input their domains of expertise, such as Databases (DB), Machine Learning (ML), User Interface (UI), etc, along with the names of any supported projects they are interested in. Once these items are selected, the front end sends this to the backend REST API endpoint. The backend then proceeds to mine the current open issues for the inputted project(s) and then classifies them by domain label. These classified issues and labels are then compared to the user’s desired domain labels to determine which issues are relevant to the user’s expertise. Once this matching process concludes, the list of relevant issues is sent back to the user on the front end. Overall, GiveMeLabeledIssues is a POC of a greater architecture to facilitate the usage of our machine learning models for issue recommendation.



### Full Product

![alt text](https://github.com/JoeyV55/GiveMeLabeledIssuesAPI/blob/master/GiveMeLabeledIssuesFull.png "POC Architecture")

The fully realized product for GiveMeLabeledIssues will use a database to store mined issues and their classified domain labels for quick access. Additionally, the full product will support both the BERT and TF-IDF models to achieve the most accurate and timely classification. This full product will be implemented directly after the POC for GiveMeLabeledIssues is completed and integrated. 


# The Team
#### Joseph Vargovich, Fabio Marcos Santos, Jacob Penney, Hanish Parthasarathy, Dr. Marco Gerosa. 

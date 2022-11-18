from csv import reader
import sys
import warnings
from django.core.management.base import BaseCommand, CommandError
from fast_bert.prediction import BertClassificationPredictor
from GiveMeLabeledIssues.BERT.databaseUtils import *

#TF-IDF imports and cleaning
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from os import path
from sklearn.metrics import hamming_loss, accuracy_score, roc_curve, auc, roc_auc_score, f1_score, multilabel_confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
import re 
from sklearn.model_selection import ShuffleSplit

#import apex
from fast_bert.prediction import BertClassificationPredictor

#Extractor import
from OSLextractor.extractor_driver import get_user_cfg
from OSLextractor.repo_extractor import conf, schema
from OSLextractor.repo_extractor.extractor import github_extractor
from OSLextractor.repo_extractor.utils import file_io_utils as file_io

MINING_PATH = '/Users/fd252/Documents/GitHub/GiveMeLabeledIssuesAPI-1/OSLextractor/docs/example_io/example_cfg.json'
PROBABILITY_THRESHOLD = .5
ISSUE_LIMIT = 1
class Command(BaseCommand):
    help = 'Populates the project issue database tables.'

    def add_arguments(self, parser):
        parser.add_argument('project')
        parser.add_argument('classifier')

    #Used for BERT  
    def buildIssueArrays(self, issuesDict):
        issueNums = list(issuesDict.keys())
        issueTexts = []
        issueTitles = []
        index = 0
        print("ISSUESDict: ", issuesDict)

        for issueNum in issueNums:
            issueTexts.append(issuesDict[issueNum]["body"] + issuesDict[issueNum]["title"])
            issueTitles.append(issuesDict[issueNum]["title"])
            index += 1

        return issueNums, issueTexts, issueTitles
    
    #Used for TF-IDF
    def buildIssueDf(self, issuesDict):
        issueNums = list(issuesDict.keys())
        issueTexts = []
        issueTitles = []

        for issueNum in issueNums:
            titleText = "" 
            bodyText = ""  
            if issuesDict[issueNum]["title"] is not None:
                titleText = issuesDict[issueNum]["title"] 
            if issuesDict[issueNum]["body"] is not None:
                bodyText = issuesDict[issueNum]["body"] 
            issueTexts.append(titleText + bodyText)
            issueTitles.append(titleText)
            
        issuesDict = {"IssueNumber" : issueNums, "IssueText" : issueTexts, "IssueTitle": issueTitles}
        issuesDf = pd.DataFrame(data=issuesDict)
        print(issuesDf.head())
        return issuesDf

    def filterLabels(self, issueLabels):
        labelStr = ""
        i = 0
        for label in issueLabels:
        # print(label)
            if label[1] >= PROBABILITY_THRESHOLD:
                labelStr += label[0]
                if i != len(issueLabels) - 1 and issueLabels[i + 1][1] >= PROBABILITY_THRESHOLD:
                    labelStr += ','
            i += 1
        return labelStr

    def classifyMinedIssuesBERT(self, issueNumbers, issueTexts, issueTitles, project):
        print("Running Bert with all model.")
        MODEL_PATH = '/Users/fd252/Documents/GitHub/GiveMeLabeledIssuesAPI-1/BERT/BERTModels/output/model_out/'
        LABEL_PATH = '/Users/fd252/Documents/GitHub/GiveMeLabeledIssuesAPI-1/BERT/BERTModels/labels/all/'
        #print(LABEL_PATH)
        predictor = BertClassificationPredictor(
                    model_path=MODEL_PATH,
                    label_path=LABEL_PATH, # location for labels.csv file
                    multi_label=True,
                    model_type='bert',
                    do_lower_case=False,
                    device=None) # set custom torch.device, defaults to cuda if available

        issues = []
        # print("ISSUETITLES: ", issueTitles)
        # print("ISSUETexts: ", issueTexts)
        # print("ISSUENUMBERS: ", issueNumbers)
        #print(domains)
        requestVals = {"issues": []}
        for i in range(0, len(issueTitles)):
            print("ISSUE CLASSIFYING: ", i)
            
            labelStr = self.filterLabels(predictor.predict(issueTexts[i]))
            issueDict = {}
            issueDict["issueTitle"] = issueTitles[i]
            issueDict["issueNumber"] = issueNumbers[i]
            issueDict["issueText"] = issueTexts[i]
        
            issueDict["issueLabels"] = labelStr
            persistToDB(issueDict, project)

            i += 1
            if i > ISSUE_LIMIT:
                break

        return requestVals        

    def clean_data(self, data_test1):
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

        def cleanHtml(sentence):
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, ' ', str(sentence))
            return cleantext

        def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
            cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
            cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
            cleaned = cleaned.strip()
            cleaned = cleaned.replace("\n"," ")
            return cleaned

        def keepAlpha(sentence):
            alpha_sent = ""
            for word in sentence.split():
                alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
                alpha_sent += alpha_word
                alpha_sent += " "
            alpha_sent = alpha_sent.strip()
            return alpha_sent

        #function pra remover palavras com menos de 3 tokens

        data_test1['IssueText'] = data_test1['IssueText'].str.lower()
        data_test1['IssueText'] = data_test1['IssueText'].apply(cleanHtml)
        data_test1['IssueText'] = data_test1['IssueText'].apply(cleanPunc)
        data_test1['IssueText'] = data_test1['IssueText'].apply(keepAlpha)
        
        return data_test1

    def remove_stop_words(self):
        stop_words = set(stopwords.words('english'))
        stop_words.update(['nan','pr','zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within','jabref','org','github','com','md','https','ad','changelog','','joelparkerhenderson','localizationupd',' localizationupd','localizationupd ','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the','Mr', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
        #stop_words.update(['i', 'me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the","Mr", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

        return re_stop_words

    def removeStopWords(self, sentence, re_stop_words):
        #global re_stop_words
        return re_stop_words.sub(" ", sentence)

    def apply_stem(self, data_test1):
        stemmer = SnowballStemmer("english")
        
        def stemming(sentence):
            stemSentence = ""
            for word in sentence.split():
                stem = stemmer.stem(word)
                stemSentence += stem
                stemSentence += " "
            stemSentence = stemSentence.strip()
            return stemSentence
        
        data_test1['IssueText'] = data_test1['IssueText'].apply(stemming)
        
        return data_test1

    #analyzing frequency of TOP 50 terms

    def analyze_top(self, data):
        docs = data['IssueText'].tolist()

        cv = CountVectorizer()
        cv_fit=cv.fit_transform(docs)

        word_list = cv.get_feature_names()   

        count_list = cv_fit.toarray().sum(axis=0)
        term_frequency = dict(zip(word_list,count_list))

        a = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True) 
        
        print('SIZE OF TERMS', len(a))
        
        top50 = a[:50]
        #df_frequency = pd.DataFrame(top50, columns =['term', 'frequency'])  

        return docs, a

    def run_tf_idf(self, data, configurationTFIDF, num_feature):
        #we need to text max_feature with 10, 20, 25, 50 
        #, max_features=num_feature
        vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range = configurationTFIDF, max_features=num_feature)
            
        tf_idf_results = vectorizer.fit_transform(data['IssueText'])

        features = vectorizer.get_feature_names()

        scores = (tf_idf_results.toarray())
        output_tf_idf = pd.DataFrame(scores)
        
        output_tf_idf = pd.concat([data['IssueNumber'], output_tf_idf], axis=1)

        return output_tf_idf

    #merging features TF-IDF with data_frame
    def merging_fast(self, data_test1, feature):
        
        data_classifier = data_test1.join(feature, lsuffix='_dc', rsuffix='_f')

        categories = data_classifier.columns.values.tolist()
        
        #del data_classifier['IssueTitle']
        #del data_classifier['IssueText']

        return data_classifier, categories

    #build the model 
    def build_model(self, test_type):
        clf = BinaryRelevance(classifier=RandomForestClassifier(criterion='entropy',max_depth= 50, min_samples_leaf= 1, min_samples_split= 3, n_estimators= 50), require_dense = [False, True])

        return clf

    def run_predictions(self, dataset, num_feature, final_columns, predictions_result, labels):

        max_f = final_columns[len(final_columns)-1]
        min_f = final_columns[0]
        print('min:',min_f,' max:',max_f)
        print('dataset:',dataset.columns)
        # each project's model was trained with different number of columns! JabRef has 720. 
        #X_test = dataset.iloc[:,2:len(dataset.columns)]
        sliced = dataset.iloc[:,4:724].copy()
        print("sliced type",type(sliced))
        print("sliced shape",sliced.shape)
        X_test = sliced.reset_index(drop=True)
        print("X_test type",type(X_test))
        print('X_test:',X_test.columns)
        print('X_text',X_test.shape)
        ids = X_test.index
        print('ids',ids)
        i = 9 #CV used from the training model
        
        # load the model to disk
        filename = predictions_result
        clf = pickle.load(open(filename, 'rb'))
        print('model loaded')

        predict = clf.predict(X_test).toarray()
        print('predictions obtained', type(predict))
        predictions = pd.DataFrame(predict, index=ids, columns=labels) # with header
       
        print('dataframe created', predictions.shape, predictions.columns)
       
        #predictions.to_csv(predictions_result+str(i)+'.csv', encoding='utf-8', sep=',')    

        data_merge = dataset.loc[:,'IssueNumber_dc':'IssueTitle']
        data_mergeDF = pd.DataFrame(data_merge)
        print('data merge created', data_mergeDF.columns)
        mergedDf = predictions.merge(data_mergeDF, left_index=True, right_index=True)
        print('mergeDF created',mergedDf.columns)
        #mergedDf.to_csv(predictions_result+'_data_merge-' + str(i) +'.csv' , encoding='utf-8', header=True, index=False , sep=',') #x = prediction
        return mergedDf

    def classifyMinedIssuesTFIDF(self, issuesDF, project):
            print("Running TFIDF with all model.")
            MODEL_PATH_TFIDF = '/Users/fd252/Documents/GitHub/GiveMeLabeledIssuesAPI-1/GiveMeLabeledIssues/TFIDF/TFIDFModels/'
            #example: /Users/fd252/Documents/GitHub/GiveMeLabeledIssuesAPI-1/GiveMeLabeledIssues/TFIDF/TFIDFModels/jabref_finalized_model.sav
            LABEL_PATH_TFIDF = '/Users/fd252/Documents/GitHub/GiveMeLabeledIssuesAPI-1/TFIDF/TFIDFModels/'
            
            #TF-IDF set up
            configurationTFIDF=(1,1) #unigrams # Used to read the right model on the file system
            stop_word = 'Yes'# stop words set up 
            test_type = "RandomForest" #ML
            size_test=0.2 #test/train # Used to read the right model on the file system

            data_test1 = self.clean_data(issuesDF)

            if stop_word == 'Yes':
                re_stop_words = self.remove_stop_words()
                data_test1['IssueText'] = data_test1['IssueText'].apply(self.removeStopWords, re_stop_words=re_stop_words)
            data = data_test1

            data_test1 = self.apply_stem(data)
            print('data_test1',data_test1.columns)

            docs, a = self.analyze_top(data_test1)
            num_feature = len(a)
            features = self.run_tf_idf(data_test1, configurationTFIDF, len(a))               

            data_classifier, categories = self.merging_fast(data_test1, features)  

            predictions_result = MODEL_PATH_TFIDF + 'jabref_finalized_model.sav'
            print("model at:", predictions_result, "project:",project)
            
            final_columns = data_classifier.columns
            
            #POC dones't have grount truth = categories. So labels are fixed:
            labels = ['Network', 'DB', 'Interpreter', 'Logging', 'Data Structure', 'i18n', 'Setup', 'Microservices', 'Test', 'IO', 'UI', 'App']

            predictions = self.run_predictions(data_classifier, num_feature, final_columns, predictions_result, labels)

            issueTitles = []
            issueNumbers = []
            issueTexts = []
            for index, row in predictions.iterrows():
                issueTitles.append(row['IssueTitle'])
                issueNumbers.append(row['IssueNumber_dc'])
                issueTexts.append(row['IssueText'])
                
            # predicions file contains the labels with 1s and 0s and the issueNumber_dc columns     

            issues = []
            # print("ISSUETITLES: ", issueTitles)
            # print("ISSUETexts: ", issueTexts)
            # print("ISSUENUMBERS: ", issueNumbers)
            #print(domains)
            requestVals = {"issues": []}
            for i in range(0, len(issueTitles)):
                print("ISSUE CLASSIFYING: ", i)
                
                labelStr = "TODO"
                issueDict = {}
                issueDict["issueTitle"] = issueTitles[i]
                issueDict["issueNumber"] = issueNumbers[i]
                issueDict["issueText"] = issueTexts[i]
            
                issueDict["issueLabels"] = labelStr
                persistToDB(issueDict, project)

                i += 1
                if i == ISSUE_LIMIT:
                    break

            return requestVals   

    def extractIssuesAndClassify(self, project, classifier):
        """Driver function for GitHub Repo Extractor."""
        
        tab: str = " " * 4

        cfg_dict: dict = get_user_cfg(MINING_PATH)
        cfg_obj = conf.Cfg(cfg_dict, schema.cfg_schema)

        # init extractor object
        print("\nInitializing extractor...")
        gh_ext = github_extractor.Extractor(project, cfg_obj)
        print(f"{tab}Extractor initialization complete!\n")

        print("Mining repo data...")
        issuesDict = gh_ext.get_repo_issues_data()
        print(f"\n{tab}Issue data complete!\n")

        if classifier == 'BERT':
            issueNumbers, issueTexts, issueTitles = self.buildIssueArrays(issuesDict)
            print("IssueNumber: " + issueNumbers[0] + " IssueText: " + issueTexts[0])
            return self.classifyMinedIssuesBERT(issueNumbers, issueTexts, issueTitles, project)
        
        #Otherwise do TF-IDF, TODO make elif when Doc2Vec is inserted.
        issueDf = self.buildIssueDf(issuesDict)
        #Run TF-IDF classification with issueDf. 
        if classifier == 'TFIDF':
            print("TFIDF - Issue: " + issueDf.iloc[0])
            return self.classifyMinedIssuesTFIDF(issueDf, project)
    
    def handle(self, *args, **options):
        valid_projects = {'jabref' : "JabRef/jabref", 'powertoys' : "microsoft/PowerToys"}
        valid_classifiers = ['BERT', 'TFIDF']

        if options['project'] not in valid_projects:
            print("Invalid project name. Valid projects are:", end=' ')
            for project in valid_projects.keys():
                print(project, end=' ')
            return
        if options['classifier'] not in valid_classifiers:
            print("Invalid project name. Valid projects are:", end=' ')
            for classifier in valid_classifiers:
                print(classifier, end=' ')
            return
        
        self.extractIssuesAndClassify(valid_projects[options['project']], options['classifier'])


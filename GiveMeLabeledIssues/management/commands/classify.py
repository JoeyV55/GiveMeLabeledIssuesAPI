import sys
import os
import warnings
from django.core.management.base import BaseCommand
from GiveMeLabeledIssues.databaseUtils import *
import numpy as np
#TF-IDF imports and cleaning
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
import re 
from sklearn.model_selection import train_test_split


#Extractor import
from OSLextractor.extractor_driver import get_user_cfg
from OSLextractor.repo_extractor import conf, schema
from OSLextractor.repo_extractor.extractor import github_extractor

MINING_PATH = (os.path.abspath(os.curdir)) + '/OSLextractor/docs/example_io/example_cfg.json'
MODEL_PATH_TFIDF = (os.path.abspath(os.curdir)) + '/GiveMeLabeledIssues/TFIDF/TFIDFModels/'
LABEL_PATH_TFIDF = (os.path.abspath(os.curdir)) + '/GiveMeLabeledIssues/TFIDF/TFIDFModels/'
ISSUE_LIMIT = 10
class Command(BaseCommand):
    help = 'Populates the project issue database tables.'

    def add_arguments(self, parser):
        parser.add_argument('project')
        parser.add_argument('classifier')

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

        word_list = cv.get_feature_names_out()   

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

        features = vectorizer.get_feature_names_out()

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
        clf = BinaryRelevance(classifier=RandomForestClassifier(criterion='entropy', max_depth= 100, min_samples_leaf= 1, min_samples_split= 3, n_estimators= 50), require_dense = [False, True])

        return clf

    def run_predictions(self, dataset, num_feature, final_columns, predictions_result, labels, projectShort):

        print("output - predictions result:",predictions_result)
        dataset.to_csv(predictions_result+'_dataset.csv' , encoding='utf-8', header=True, index=True , sep=',') #x = dataset
        idx = dataset.index
        print("dataset idx:",idx)

        max_f = final_columns[len(final_columns)-1]
        min_f = final_columns[0]
        print('min:',min_f,' max:',max_f)
        print('dataset cols:',dataset.columns)
        # each project's model was trained with different number of columns! JabRef has 720. 
        #X_test = dataset.iloc[:,2:len(dataset.columns)]
        if projectShort == 'jabref':
            sliced = dataset.iloc[:,4:724].copy()
        elif projectShort == 'powertoys':
                sliced = dataset.iloc[:,4:2058].copy()

        print("sliced type",type(sliced))
        print("sliced shape",sliced.shape)
        X_test = sliced.reset_index(drop=True)
        print("X_test type",type(X_test))
        print('X_test:',X_test.columns)
        print('X_text',X_test.shape)
        ids = X_test.index
        print("X_test ids:",ids)
        
        i = 9 #CV used from the training model
        
        # load the model to disk
        filename = predictions_result
        print('filename:',filename)
        clf = pickle.load(open(filename, 'rb'))
        print('model loaded')

        predict = clf.predict(X_test).toarray()
        print('predictions obtained', type(predict))
        predictions = pd.DataFrame(predict, index=ids, columns=labels) # with header
       
        print('dataframe created', predictions.shape, predictions.columns)
       
        predictions.to_csv(predictions_result+str(i)+'.csv', encoding='utf-8', sep=',')    

        data_merge = dataset.loc[:,'IssueNumber_dc':'IssueTitle']
        data_mergeDF = pd.DataFrame(data_merge)
        print('data merge created', data_mergeDF.columns)
        mergedDf = predictions.merge(data_mergeDF, left_index=True, right_index=True)
        print('mergeDF created',mergedDf.columns)
        mergedDf.to_csv(predictions_result+'_data_merge-' + str(i) +'.csv' , encoding='utf-8', header=True, index=False , sep=',') #x = prediction
        return mergedDf

    def run_predictions_train_test(self, dataset, num_feature, final_columns, predictions_result, labels, projectShort):

        max_f = final_columns[len(final_columns)-1]
        min_f = final_columns[0]
        print('min:',min_f,' max:',max_f)
        print('dataset cols:',dataset.columns)
        print("dataset:", dataset)
        test_i=dataset.index
        # each project's model was trained with different number of columns! JabRef has 720. 
        #X_test = dataset.iloc[:,2:len(dataset.columns)]
        if projectShort == 'jabref':
            sliced = dataset.iloc[:,4:724].copy()
        elif projectShort == 'powertoys':
                sliced = dataset.iloc[:,4:2058].copy()
        id_corpus_test = dataset.iloc[:,0:2].copy()
        print("id_corpus_test cols:",id_corpus_test.columns)
        print("id_corpus_test:",id_corpus_test.shape)

        print("sliced type",type(sliced))
        print("sliced shape",sliced.shape)
        X_test = sliced.reset_index(drop=True)
        #print("X_test type",type(X_test))
        X_test.columns.astype(str)
        print('X_test cols:',X_test.columns)
        print('X_test',X_test.shape)
        ids = X_test.index
        print('ids',ids)
        
        if projectShort == 'jabref': 
 
            y_test = pd.DataFrame(columns=['Network', 'DB', 'Interpreter', 'Logging', 'Data Structure', 'i18n', 'Setup', 'Microservices', 'Test', 'IO', 'UI', 'App'])
            print("populating y_test")
            print("y_test cols:",y_test.columns)

            test = pd.concat([y_test,X_test], axis=1)
            test['Network'] = 0
            test['DB'] = 0
            test['Interpreter'] = 0
            test['Logging'] = 0
            test['Data Structure'] = 0
            test['i18n'] = 0
            test['Setup'] = 0
            test['Microservices'] = 0
            test['Test'] = 0
            test['IO'] = 0
            test['UI'] = 0
            test['App'] = 0

            print("test:",test)
            print("Test columns:",test.columns)

        elif projectShort == 'powertoys': 

            y_test = pd.DataFrame(columns=['APM', 'Interpreter', 'Logging', 'Data Structure', 'i18n', 'Setup','Logic', 'Microservices', 'Test', 'Search', 'UI', 'Parser', 'App'])
            print("populating y_test")
            print("y_test cols:",y_test.columns)

            #print("y_test",y_test)
            test = pd.concat([y_test,X_test], axis=1)
            test['APM'] = 0
            test['Interpreter'] = 0
            test['Logging'] = 0
            test['Data Structure'] = 0
            test['i18n'] = 0
            test['Setup'] = 0
            test['Logic'] = 0
            test['Microservices'] = 0
            test['Test'] = 0
            test['Search'] = 0
            test['UI'] = 0
            test['Parser'] = 0
            test['App'] = 0

            print("test:",test)
            print("Test columns:",test.columns)

        train_csv = pd.read_csv(predictions_result)
        print("Train_csv columns:",train_csv.columns)
        del train_csv['prNumber']
        del train_csv['corpus']
        train_csv.rename(columns={'issueNumberissueNumber': 'IssueNumber_dc'}, inplace=True)
        #del train[0]    
        train_csv.drop(train_csv.columns[0],axis=1,inplace=True)
        print("train_csv",train_csv.columns)
        print("train_csv",train_csv.shape)
        id_corpus_train = train_csv.iloc[:,0:1].copy()
        id_corpus_train ['IssueText'] = np.nan
        id_corpus_train ['IssueTitle'] = np.nan
        print("id_corpus_traincols:",id_corpus_train.columns)
        print("id_corpus_train:",id_corpus_train.shape)
        id_corpus = pd.concat([id_corpus_train,id_corpus_test], axis=0)
        print("id_corpus cols:",id_corpus.columns)
        print("id_corpus:",id_corpus.shape)

        if projectShort == 'jabref':
            X_train = train_csv.iloc[:,13:734].copy() #720 tfidf features + issue number = 721 tfidf powertoys training dataset + 12 labels + 1 = 734
            y_train = train_csv.iloc[:,1:13] 
        elif projectShort == 'powertoys':
                X_train = train_csv.iloc[:,14:2068].copy() #2053 tfidf features + issue number = 2054 tfidf powertoys training dataset + 13 labels + 1 = 2068
                y_train = train_csv.iloc[:,1:14] 

        X_train.rename(columns={'issueNumberissueNumber.1': '0'}, inplace=True)
        print("X_train data:",X_train)
        #X_train.columns.astype(int)
        X_train.columns = np.arange(len(X_train.columns))
        print("X_train cols",X_train.columns)
        print("X_train",X_train.shape)
        print("y_train cols",y_train.columns)
        print("y_train",y_train.shape)
        print("y_train",y_train)
        train = pd.concat([y_train,X_train], axis=1)
        dataset_fit = pd.concat([train,test], axis=0)
        #idx = dataset_fit.index
        print('dataset_fit cols:',dataset_fit.columns)
        print('dataset_fit:',dataset_fit.shape)
        print("dataset_fit:",dataset_fit)
        if projectShort == 'jabref':
            ds_X = dataset_fit.iloc[:,13:724].copy()
            ds_y = dataset_fit.iloc[:,0:12] #12 labels
        elif projectShort == 'powertoys':
                ds_X = dataset_fit.iloc[:,14:2068].copy() #2054 + 13 labels + 1
                ds_y = dataset_fit.iloc[:,0:13] 

        print("ds_X",ds_X.shape)
        print("ds_y cols",ds_y.columns)
        print("ds_y",ds_y.shape)

        X = pd.concat([X_train,X_test], axis=0, 
                  ignore_index = True)
        y = pd.concat([y_train,
        y_test], axis=0)
        print("X cols",X.columns)
         
        print("X",X.shape)
        print("y cols",y.columns)
        print("y",y.shape)
        
        X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(ds_X, ds_y, test_size=0.8, random_state=42)
        idx = y_test_n.index
        print('idx',idx)
        clf = BinaryRelevance(classifier=RandomForestClassifier(criterion='entropy',max_depth= 50, min_samples_leaf= 1, min_samples_split= 3, n_estimators= 50), require_dense = [False, True])
        clf.fit(X_train_n,y_train_n)

        predict = clf.predict(X_test_n).toarray()
        print('predictions obtained', type(predict))
        #predictions = pd.DataFrame(predict, index=ids, columns=labels) # with header
        predictions = pd.DataFrame(predict, index=idx, columns=labels) # with header
       
        print('dataframe created', predictions.shape, predictions.columns)
       
        predictions.to_csv(predictions_result+'.csv', encoding='utf-8', sep=',')    

        data_merge = id_corpus.loc[:,'IssueNumber_dc':'IssueTitle']
        data_mergeDF = pd.DataFrame(data_merge)
        print('data merge created', data_mergeDF.columns)
        mergedDf = predictions.merge(data_mergeDF, left_index=True, right_index=True)
        print('mergeDF cols',mergedDf.columns)
        print('mergeDF shape',mergedDf.shape)
        #mergedDf.dropna(inplace=True) 
        mergedDf = mergedDf[mergedDf['IssueText'].notna()]
        print('mergeDF shape after removed NaN',mergedDf.shape)
        mergedDf.to_csv(predictions_result+'_data_merge_POC.csv' , encoding='utf-8', header=True, index=False , sep=',') #x = prediction
        return mergedDf


    def classifyMinedIssuesTFIDF(self, issuesDF, project, projectShort):
            print("Running TFIDF with all model.")
            
            #TF-IDF set up
            configurationTFIDF=(1,1) #unigrams # Used to read the right model on the file system
            stop_word = 'Yes'# stop words set up 
            test_type = "RandomForest" #ML
            size_test=0.2 #test/train # Used to read the right model on the file system

            data_test1 = self.clean_data(issuesDF)

            if stop_word == 'Yes':
                re_stop_words = self.remove_stop_words()
                data_test1['IssueText'] = data_test1['IssueText'].apply(self.removeStopWords, re_stop_words=re_stop_words)

            data_test1 = self.apply_stem(data_test1)
           # print('data_test1',data_test1.columns)

            docs, a = self.analyze_top(data_test1)
            num_feature = len(a)
            features = self.run_tf_idf(data_test1, configurationTFIDF, len(a))               

            data_classifier, categories = self.merging_fast(data_test1, features)  

            predictions_result = MODEL_PATH_TFIDF + projectShort + '.csv'
            #predictions_result = MODEL_PATH_TFIDF + projectShort + '_model.sav'
            print("model at:", predictions_result, "project:",project)
            
            final_columns = data_classifier.columns
            print("final columns:", final_columns)

            #POC dones't have grount truth = categories. So labels are fixed:
            if projectShort == 'jabref': 
                labels = ['Network', 'DB', 'Interpreter', 'Logging', 'Data Structure', 'i18n', 'Setup', 'Microservices', 'Test', 'IO', 'UI', 'App']
            elif projectShort == 'powertoys':
                labels = ['APM','Interpreter','Logging','Data Structure','i18n','Setup','Logic','Microservices','Test','Search','UI','Parser','App']

            #predictions = self.run_predictions(data_classifier, num_feature, final_columns, predictions_result, labels, projectShort)
            predictions = self.run_predictions_train_test(data_classifier, num_feature, final_columns, predictions_result, labels, projectShort)

            issueTitles = []
            issueNumbers = []
            issueTexts = []
            for index, row in predictions.iterrows():
                issueTitles.append(row['IssueTitle'])
                issueNumbers.append(row['IssueNumber_dc'])
                issueTexts.append(row['IssueText'])
                
            # predicions file contains the labels with 1s and 0s and the issueNumber_dc columns     

            issues = []
            labelStr = ""
            # print("ISSUETITLES: ", issueTitles)
            # print("ISSUETexts: ", issueTexts)
            # print("ISSUENUMBERS: ", issueNumbers)
            #print(domains)
            print("\n==========================\nPREDICTIONS\n===========================\n", predictions)
            #print(predictions.columns)
            requestVals = {"issues": []}
            i = 0
            for index, row in predictions.iterrows():
                labelStr = ""
                labelCount = 0
              #  print("ROW: ", i)
                for column in predictions.columns:
                   # print("COLUMN TYPE: ", type(column))
                    if column.strip() == 'IssueNumber_dc': 
                       # print("Ënding row")
                        break

                   # print("Column: ", column)
                   # print("Value: ", row[column])
                    if row[column] == 1.0:
                        labelStr += column.strip().replace(" ", '.')
                        labelStr += ","
                        labelCount += 1
                if labelCount > 1:
                    labelStr = labelStr[:-1]

               # print("ISSUE: ", i, " LABELS: ", labelStr)
                issueDict = {}
                issueDict["issueTitle"] = row["IssueTitle"]
                issueDict["issueNumber"] = row["IssueNumber_dc"]
                issueDict["issueText"] = row["IssueText"]
            
                issueDict["issueLabels"] = labelStr
                persistToDB(issueDict, project)

                #i += 1
                #if i == ISSUE_LIMIT:
                #    break

            return requestVals   

    def extractIssuesAndClassify(self, project, projectShort, classifier):
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

       # TODO make elif when Doc2Vec is inserted.
        issueDf = self.buildIssueDf(issuesDict)
        #Run TF-IDF classification with issueDf. 
        if classifier == 'TFIDF':
            print("TFIDF - Issue: " + issueDf.iloc[0])
            return self.classifyMinedIssuesTFIDF(issueDf, project, projectShort)
    
    def handle(self, *args, **options):
        valid_projects = {'jabref' : "JabRef/jabref", 'powertoys' : "microsoft/PowerToys"}
        valid_classifiers = ['TFIDF']

        if options['project'] not in valid_projects:
            print("Invalid project name. Valid projects are:", end=' ')
            for project in valid_projects.keys():
                print(project, end=' ')
            return
        if options['classifier'] not in valid_classifiers:
            print("Invalid classifier name. Valid classifiers are:", end=' ')
            for classifier in valid_classifiers:
                print(classifier, end=' ')
            return
        
        self.extractIssuesAndClassify(valid_projects[options['project']], options['project'], options['classifier'])


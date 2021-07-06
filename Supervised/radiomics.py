#- imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,ShuffleSplit,cross_val_score,GridSearchCV,StratifiedKFold,\
                                    cross_val_predict

from sklearn.metrics import confusion_matrix,make_scorer,accuracy_score,matthews_corrcoef,roc_auc_score,\
                            recall_score,precision_score,plot_confusion_matrix,f1_score

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,\
                             BaggingClassifier,AdaBoostClassifier,ExtraTreesClassifier

from sklearn.feature_selection import mutual_info_classif,chi2,SelectKBest,RFECV,f_classif,SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from skrebate import ReliefF

from mlxtend.feature_selection import SequentialFeatureSelector

from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier, Pool

from lightgbm.sklearn import LGBMClassifier

import warnings

warnings.filterwarnings("ignore")


# Preprocessing

data = pd.read_csv("pulmoner_data.csv")

x = data.iloc[:,2:]
y = data[["Hastalik"]]
X = (x - np.min(x))/(np.max(x) - np.min(x))

y.loc[y["Hastalik"] == "No PF", "Hastalik"] = 0
y.loc[y["Hastalik"] == "PF", "Hastalik"] = 0
y.loc[y["Hastalik"] == "Control", "Hastalik"] = 1

y["Hastalik"] = y["Hastalik"].astype("int64")

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


classificationType = "Control/Pf,NonPf"
cv = 10
max_feature_num = 10

classificationList = list()
featureSelectionList = list()
algorithmList = list()
numberOfFeaturesList = list()
trainAccScoreList = list()
trainMccScoreList = list()
trainAucScoreList = list()
trainSenScoreList = list()
trainSpeScoreList = list()
trainF1ScoreList = list()
trainTPList = list()
trainFPList = list()
trainFNList = list()
trainTNList = list()
trainPPVList = list()
trainNPVList = list()
testAccScoreList = list()
testMccScoreList = list()
testAucScoreList = list()
testSenScoreList = list()
testSpeScoreList = list()
testF1ScoreList = list()
testTPList = list()
testFPList = list()
testFNList = list()
testTNList = list()
testPPVList = list()
testNPVList = list()
features = list()

#-Functions
def addToList(cla,fs,alg,model_scores,nof,f):
    classificationList.append(cla)
    featureSelectionList.append(fs)
    algorithmList.append(alg)
    numberOfFeaturesList.append(nof)
    features.append(f)
    
    trainAccScoreList.append(model_scores[0])
    trainMccScoreList.append(model_scores[1])
    trainAucScoreList.append(model_scores[2])
    trainSenScoreList.append(model_scores[3])
    trainSpeScoreList.append(model_scores[4])
    trainF1ScoreList.append(model_scores[5])
    trainPPVList.append(model_scores[6])
    trainNPVList.append(model_scores[7])
    
    train_conf = model_scores[8]
    tn, fp, fn, tp = train_conf.ravel()
        
    trainTPList.append(tp)
    trainFPList.append(fp)
    trainFNList.append(fn)
    trainTNList.append(tn)
    
    
    testAccScoreList.append(model_scores[9])
    testMccScoreList.append(model_scores[10])
    testAucScoreList.append(model_scores[11])
    testSenScoreList.append(model_scores[12])
    testSpeScoreList.append(model_scores[13])
    testF1ScoreList.append(model_scores[14])
    testPPVList.append(model_scores[15])
    testNPVList.append(model_scores[16])
    
    test_conf = model_scores[17]
    tn, fp, fn, tp = test_conf.ravel()
        
    testTPList.append(tp)
    testFPList.append(fp)
    testFNList.append(fn)
    testTNList.append(tn)

def runModel(model,X,Y,X_test,y_test,cv):
    cvAccScores = cross_val_score(model, X=X, y=Y, cv=cv, scoring="accuracy")
    cvMccScores = cross_val_score(model, X=X, y=Y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    cvAucScores = cross_val_score(model, X=X, y=Y, cv=cv, scoring="roc_auc")
    cvRecScores = cross_val_score(model, X=X, y=Y, cv=cv, scoring="recall")
    cvF1Scores = cross_val_score(model, X=X, y=Y, cv=cv, scoring="f1")
    
    meanCvAccScore = np.mean(cvAccScores)
    meanCvMccScore = np.mean(cvMccScores)
    meanCvAucScore = np.mean(cvAucScores)
    meanCvSenScore = np.mean(cvRecScores)
    meanCvF1Score = np.mean(cvF1Scores)
    
    yPred = cross_val_predict(model,X,Y,cv=cv)
    train_conf = confusion_matrix(Y,yPred)
    
    tn, fp, fn, tp = train_conf.ravel()

    try:
        meanCvSpeScore = tn/(tn+fp)#tn/tn+fp
    except Exception as Ex:
        meanCVSpeScore = 0
        print(Ex)
    
    try:
        meanCvPpvScore = tp/(tp+fp)#tp/(tp+fp)
    except Exception as Ex:
        meanCVPpvScore = 0
        print(Ex)
    
    try:
        meanCvNpvScore = tn/(tn+fn)
    except Exception as Ex:
        meanCVNpvScore = 0
        print(Ex)
    
    #Test
    
    yPredTest = cross_val_predict(model,X_test,y_test,cv=cv)
    
    testAccScore = accuracy_score(y_test, yPredTest)
    testMccScore = matthews_corrcoef(y_test, yPredTest)
    testAucScore = roc_auc_score(y_test, yPredTest)
    testSenScore = recall_score(y_test, yPredTest)
    testF1Score = f1_score(y_test, yPredTest)
    
    test_conf = confusion_matrix(y_test,yPredTest)
    
    tn, fp, fn, tp = test_conf.ravel()

    try:
        testSpeScore = tn/(tn+fp)#tn/tn+fp
    except Exception as Ex:
        testSpeScore = 0
        print(Ex)
        
    try:
        testPpvScore = tp/(tp+fp)#tp/(tp+fp)
    except Exception as Ex:
        testPpvScore = 0
        print(Ex)
    
    try:
        testNpvScore = tn/(tn+fn)
    except Exception as Ex:
        testNpvScore = 0
        print(Ex)

    return (meanCvAccScore,meanCvMccScore,meanCvAucScore,meanCvSenScore,meanCvSpeScore\
            ,meanCvPpvScore,meanCvNpvScore,meanCvF1Score,train_conf,testAccScore,testMccScore\
            ,testAucScore,testSenScore,testSpeScore,testPpvScore,testNpvScore,testF1Score,test_conf)

def saveResults():
    df = pd.DataFrame({
        "Classification Type" : classificationList,
        "Feature Selection Method" : featureSelectionList,
        "Number of Features" : numberOfFeaturesList,
        "ML Algorithm" : algorithmList,
        "Train Accuracy" : trainAccScoreList,
        "Train MCC" : trainMccScoreList,
        "Train AUC" : trainAucScoreList,
        "Train Sensitivity" : trainSenScoreList,
        "Train Specifity" : trainSpeScoreList,
        "Train PPV" : trainPPVList,
        "Train NPV" : trainNPVList,
        "Train F1" : trainF1ScoreList,
        "Train TP" : trainTPList,
        "Train FP" : trainFPList,
        "Train FN" : trainFNList,
        "Train TN" : trainTNList,
        "Test Accuracy" : testAccScoreList,
        "Test MCC" : testMccScoreList,
        "Test AUC" : testAucScoreList,
        "Test Sensitivity" : testSenScoreList,
        "Test Specifity" : testSpeScoreList,
        "Test PPV" : testPPVList,
        "Test NPV" : testNPVList,
        "Test F1" : testF1ScoreList,
        "Test TP" : testTPList,
        "Test FP" : testFPList,
        "Test FN" : testFNList,
        "Test TN" : testTNList,
        "Features" : features
    })
    df.to_csv("Control_Pf-NonPfSonuclar2.csv")

def printScores(model_scores):
    trainString = """\t\t\tTrain
                \t\tAccuracy : {}
                \t\tMcc : {}
                \t\tAuc : {}
                \t\tSensitivity : {}
                \t\tSpecifity : {}
                \t\tF1 : {}
                \t\tPPV : {}
                \t\tNPV : {}
                \t\tTP : {} \tFP : {}
                \t\tFN : {} \tTN : {}
                \tTest
                \t\tAccuracy : {}
                \t\tMcc : {}
                \t\tAuc : {}
                \t\tSensitivity : {}
                \t\tSpecifity : {}
                \t\tF1 : {}
                \t\tPPV : {}
                \t\tNPV : {}
                \t\tTP : {}  \tFP : {}
                \t\tFN : {}  \tTN : {}""".format(model_scores[0],model_scores[1],\
            model_scores[2],model_scores[3],model_scores[4],model_scores[5],\
            model_scores[6],model_scores[7],*model_scores[8].ravel(),\
            model_scores[9],model_scores[10],model_scores[11],model_scores[12],\
            model_scores[13],model_scores[14],model_scores[15],model_scores[16],\
            *model_scores[17].ravel())

    print(trainString)

#-Logistic Regression
print("\n\n\n"+classificationType)

ml = "Logistic Regression"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = {
    "max_iter" : [10,20,30,40,50,60,70,80,90,100],
}

fselection="None"
print("\t\t"+fselection)
model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X_train.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = LogisticRegression()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = LogisticRegression()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=LogisticRegression(),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = LogisticRegression()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = LogisticRegression()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(LogisticRegression(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(LogisticRegression(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(LogisticRegression())
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = LogisticRegression()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

#-KNN
print("\n\n\n"+classificationType)

ml = "KNN"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = { 
    "n_neighbors" : [5,10,15,20,25],
    "leaf_size" : [2,5,10,15,20,25]
}


fselection="None"
print("\t\t"+fselection)
model = KNeighborsClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = KNeighborsClassifier()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = KNeighborsClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = KNeighborsClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = KNeighborsClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = KNeighborsClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = KNeighborsClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = KNeighborsClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = KNeighborsClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(KNeighborsClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = KNeighborsClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(KNeighborsClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = KNeighborsClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

#-SVM
print("\n\n\n"+classificationType)

ml = "SVM"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = {
    "C" : [0.001, 0.1, 1, 10],  
    "gamma" : [0.0001, 0.001, 0.01, 0.1, 1], 
    "kernel" : ["linear", "poly", "rbf", "sigmoid"]
}



fselection="None"
print("\t\t"+fselection)
model = SVC()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = SVC()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = SVC()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = SVC()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = SVC()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = SVC()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = SVC()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = SVC()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = SVC()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(SVC(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = SVC()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(SVC(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = SVC()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()


#-Naive Bayes
print("\n\n\n"+classificationType)

ml = "Naive Bayes"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

fselection="None"
print("\t\t"+fselection)
model = GaussianNB()
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,"Naive Bayes",model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = GaussianNB()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = GaussianNB()
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = GaussianNB()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = GaussianNB()
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = GaussianNB()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = GaussianNB()
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = GaussianNB()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = GaussianNB()
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(GaussianNB(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = GaussianNB()
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(GaussianNB(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = GaussianNB()
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()


#-Decision Tree
print("\n\n\n"+classificationType)

ml = "Decision Tree"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = {
    "max_depth" : [2,4,6,8],
    "max_leaf_nodes" : [20, 50, 80],
    "min_samples_split": [2, 3, 4]
}

fselection="None"
print("\t\t"+fselection)
model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = DecisionTreeClassifier()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = DecisionTreeClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=DecisionTreeClassifier(),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = DecisionTreeClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = DecisionTreeClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(DecisionTreeClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(DecisionTreeClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(DecisionTreeClassifier())
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = DecisionTreeClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

#-Random Forest
print("\n\n\n"+classificationType)

ml = "Random Forest"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = { 
    "n_estimators": [20,40,80],
    "max_depth" : [2,4,6],
}

fselection="None"
print("\t\t"+fselection)
model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = RandomForestClassifier()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = RandomForestClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=RandomForestClassifier(),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = RandomForestClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = RandomForestClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(RandomForestClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(RandomForestClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(RandomForestClassifier())
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = RandomForestClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

#-Bagging
print("\n\n\n"+classificationType)

ml = "Bagging"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = {
    "base_estimator" : [SVC(),
                       KNeighborsClassifier(n_neighbors=3),
                       GradientBoostingClassifier(),
                       DecisionTreeClassifier(),
                       BaggingClassifier()],
    "n_estimators" : [10,20,30,40],
    "max_samples" : [0.05, 0.1, 0.2, 0.5]
}

fselection="None"
print("\t\t"+fselection)
model = BaggingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = BaggingClassifier()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = BaggingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = BaggingClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = BaggingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = BaggingClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = BaggingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = BaggingClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = BaggingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(BaggingClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = BaggingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(BaggingClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = BaggingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

#-AdaBoost
print("\n\n\n"+classificationType)

ml = "AdaBoost"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = { 
    "base_estimator" : [SVC(),
                       KNeighborsClassifier(),
                       GradientBoostingClassifier(),
                       DecisionTreeClassifier(),
                       RandomForestClassifier()],
    "n_estimators": [10,20,50],
    "learning_rate" : [0.1, 0.05, 0.01, 0.005]    
}

fselection="None"
print("\t\t"+fselection)
model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = AdaBoostClassifier()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = AdaBoostClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=AdaBoostClassifier(),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = AdaBoostClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = AdaBoostClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(AdaBoostClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(AdaBoostClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(AdaBoostClassifier())
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = AdaBoostClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

#-Gradient Boost
print("\n\n\n"+classificationType)

ml = "Gradient Boost"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = { 
    "n_estimators": [20,50,100],
    "max_depth" : [3,5,7],
    "learning_rate" : [0.1, 0.05, 0.01, 0.005],
    "subsample" : [0.6,0.8,1.0] 
}

fselection="None"
print("\t\t"+fselection)
model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = GradientBoostingClassifier()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = GradientBoostingClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=GradientBoostingClassifier(),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = GradientBoostingClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = GradientBoostingClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(GradientBoostingClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(GradientBoostingClassifier(), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(GradientBoostingClassifier())
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = GradientBoostingClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()


#-XGBoost
print("\n\n\n"+classificationType)

ml = "XGBoost"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = {
    "n_estimators" : range(50,300,50),
    "learning_rate" : [0.1, 0.05, 0.01],
    "max_depth" : [2, 3, 4, 5, 6, 8]
}

fselection="None"
print("\t\t"+fselection)
model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = XGBClassifier(verbosity=0)
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = XGBClassifier(verbose=0)
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=XGBClassifier(verbose=0),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = XGBClassifier(verbose=0)
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = XGBClassifier(verbose=0)
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFS
fselection="SFS"
print("\t\t"+fselection)
sfs = SequentialFeatureSelector(XGBClassifier(verbose=0), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sfs.fit_transform(X_train.values,y_train.values.ravel())
x = sfs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sfs.k_feature_names_).astype("int")])

model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--SFFS
fselection="SFFS"
print("\t\t"+fselection)
sffs = SequentialFeatureSelector(XGBClassifier(verbose=0), 
           k_features=(1,max_feature_num), 
           forward=True,
           floating=True, 
           verbose=0,
           scoring='accuracy',
           cv=cv)

X = sffs.fit_transform(X_train.values,y_train.values.ravel())
x = sffs.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[np.array(sffs.k_feature_names_).astype("int")])

model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(XGBClassifier(verbose=0))
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = XGBClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()


#-CatBoost
print("\n\n\n"+classificationType)

ml = "CatBoost"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = {
    "learning_rate" : [0.1, 0.05, 0.01],
    "iterations" : [20,50,100]
}

fselection="None"
print("\t\t"+fselection)
model = CatBoostClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = CatBoostClassifier(verbose=0)
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = CatBoostClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = CatBoostClassifier(verbose=0)
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = CatBoostClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=CatBoostClassifier(verbose=0),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = CatBoostClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = CatBoostClassifier(verbose=0)
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = CatBoostClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = CatBoostClassifier(verbose=0)
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = CatBoostClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(CatBoostClassifier(verbose=0))
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = CatBoostClassifier(verbose=0)
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

#-LightGBM
print("\n\n\n"+classificationType)

ml = "LightGBM"

print("\t"+ml)

X=X_train
Y=y_train
x=X_test
y=y_test

params = {
    "learning_rate" : [0.1, 0.05, 0.01, 0.005],
    "n_estimators" : [8,16,24],
    "num_leaves": [6,8,12,16]
}

fselection="None"
print("\t\t"+fselection)
model = LGBMClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X.values,Y.values.ravel().astype("int64"))
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],"All")
printScores(model_scores)


#--Information Gain
fselection="Information Gain"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(mutual_info_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = LGBMClassifier()
    acc_score = np.mean(cross_val_score(model, X=x, y=y.values.ravel(), cv=cv, scoring="accuracy"))

    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
    

selectK = SelectKBest(mutual_info_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = LGBMClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,best_n_feature,best_features)
printScores(model_scores)

#--Chi2
fselection="Chi2"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(chi2, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    model = LGBMClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))   
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(chi2, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns] 
x = X_test.iloc[:,importantColumns] 
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)
model = LGBMClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--RFE
fselection="RFE"
print("\t\t"+fselection)

rfe = RFECV(estimator=LGBMClassifier(),cv=cv)
rfe.fit(X_train.values,y_train.values.ravel())
importantColumns = rfe.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",len(importantColumns))
best_features = list(X.columns)

model = LGBMClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ANOVA
fselection="ANOVA"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0
for n_feature in range(1,max_feature_num):
    selectK = SelectKBest(f_classif, k=n_feature)
    selectK.fit(X_train.values, y_train.values.ravel())
    importantColumns = selectK.get_support(indices=True)
    X = X_train.iloc[:,importantColumns]
    
    model = LGBMClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(),cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
selectK = SelectKBest(f_classif, k=best_n_feature)
selectK.fit(X_train.values, y_train.values.ravel())
importantColumns = selectK.get_support(indices=True)
X = X_train.iloc[:,importantColumns]
x = X_test.iloc[:,importantColumns]
print("\t\tFeatures : ",best_n_feature)
best_features = list(X.columns)

model = LGBMClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)


#--ReliefF
fselection="ReliefF"
print("\t\t"+fselection)
max_f_score = 0
acc_score = 0
best_n_feature = 0

for n_feature in range(1,max_feature_num):
    relief = ReliefF(n_features_to_select=n_feature, n_neighbors=20)
    X = relief.fit_transform(X_train.values,y_train.values.ravel())
    
    model = LGBMClassifier()
    acc_score = np.mean(cross_val_score(model, X=X, y=Y.values.ravel(), cv=cv, scoring="accuracy"))
    
    if acc_score > max_f_score:
        max_f_score = acc_score
        best_n_feature = n_feature
        
relief = ReliefF(n_features_to_select=best_n_feature, n_neighbors=20)
X = relief.fit_transform(X_train.values,y_train.values.ravel())
x = relief.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",best_n_feature)
best_features = list(X_train.columns[relief.top_features_[:best_n_feature]])

model = LGBMClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

#--Embedded
fselection="Embedded"
print("\t\t"+fselection)

embeded_rf_selector = SelectFromModel(LGBMClassifier())
X = embeded_rf_selector.fit_transform(X_train.values, y_train.values.ravel())
x = embeded_rf_selector.fit_transform(X_test.values,y_test.values.ravel())
print("\t\tFeatures : ",X.shape[1])
best_features = list(X_train.columns[embeded_rf_selector.get_support()])

model = LGBMClassifier()
gridSCV = GridSearchCV(model,params,cv=cv)
gridSCV.fit(X,Y.values.ravel())
model = gridSCV.best_estimator_
model_scores = runModel(model,X,Y.values.ravel(),x,y.values.ravel(),cv)
addToList(classificationType,fselection,ml,model_scores,X.shape[1],best_features)
printScores(model_scores)

saveResults()

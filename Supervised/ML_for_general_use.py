import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier


warnings.filterwarnings("ignore")

#Define Machine Learning Models
def FineTree(X_train,y_train,Y_of_plot):
    FineTree = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=100)
    FineTree.fit(X_train,y_train)
    y_pred = FineTree.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)


def MediumTree(X_train,y_train,Y_of_plot):
    MediumTree = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=20)
    MediumTree.fit(X_train,y_train)
    y_pred = MediumTree.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def CoarseTree(X_train,y_train,Y_of_plot):
    CoarseTree = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=2)
    CoarseTree.fit(X_train,y_train)
    y_pred = CoarseTree.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)
    
def LineerDiscriminant(X_train,y_train,Y_of_plot):
    LineerDiscriminant = LinearDiscriminantAnalysis()
    LineerDiscriminant.fit(X_train,y_train)
    y_pred = LineerDiscriminant.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)
    
def QuadraticDiscriminant(X_train,y_train,Y_of_plot):
    QuadraticDiscriminant = QuadraticDiscriminantAnalysis()
    QuadraticDiscriminant.fit(X_train,y_train)
    y_pred = QuadraticDiscriminant.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def LogisticRegressionM(X_train,y_train,Y_of_plot):
    LogisticRegressionM = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    LogisticRegressionM.fit(X_train,y_train)
    y_pred = LogisticRegressionM.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def GaussianNaiveBayes(X_train,y_train,Y_of_plot):
    GaussianNaiveBayes = GaussianNB()
    GaussianNaiveBayes.fit(X_train,y_train)
    y_pred = GaussianNaiveBayes.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def LinearSVM(X_train,y_train,Y_of_plot):
    LinearSVM = LinearSVC()
    LinearSVM.fit(X_train,y_train)
    y_pred = LinearSVM.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def QuadricSVM(X_train,y_train,Y_of_plot):
    QuadricSVM = SVC(kernel='poly',degree=2)
    QuadricSVM.fit(X_train,y_train)
    y_pred = QuadricSVM.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def CubicSVM(X_train,y_train,Y_of_plot):
    CubicSVM = SVC(kernel='poly',degree=3)
    CubicSVM.fit(X_train,y_train)
    y_pred = CubicSVM.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def GaussianSVM(X_train,y_train,Y_of_plot):
    GaussianSVM = SVC(kernel='rbf')
    GaussianSVM.fit(X_train,y_train)
    y_pred = GaussianSVM.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def FineKNN(X_train,y_train,Y_of_plot):
    FineKNN = KNeighborsClassifier(n_neighbors=1)
    FineKNN.fit(X_train,y_train)
    y_pred = FineKNN.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def MediumKNN(X_train,y_train,Y_of_plot):
    MediumKNN = KNeighborsClassifier(n_neighbors=10)
    MediumKNN.fit(X_train,y_train)
    y_pred = MediumKNN.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

# if samples are less than 100 doesn't work
# def CoarseKNN(X_train,y_train,Y_of_plot):
#     CoarseKNN = KNeighborsClassifier(n_neighbors=100)
#     CoarseKNN.fit(X_train,y_train)
#     y_pred = CoarseKNN.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
#     acc = (accuracy_score(y_test,y_pred))
#     Y_of_plot.append(acc)

def CubicKNN(X_train,y_train,Y_of_plot):
    CubicKNN = KNeighborsClassifier(n_neighbors=10,p=3)
    CubicKNN.fit(X_train,y_train)
    y_pred = CubicKNN.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def WeightedKNN(X_train,y_train,Y_of_plot):
    WeightedKNN = KNeighborsClassifier(n_neighbors=10,weights=('distance'))
    WeightedKNN.fit(X_train,y_train)
    y_pred = WeightedKNN.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def XGBoost(X_train,y_train,Y_of_plot):
    XGBoost = XGBClassifier()
    XGBoost.fit(X_train,y_train)
    y_pred = XGBoost.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def AdaBoost(X_train,y_train,Y_of_plot):
    AdaBoost = AdaBoostClassifier(learning_rate=0.1)
    AdaBoost.fit(X_train,y_train)
    y_pred = AdaBoost.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def Bagging(X_train,y_train,Y_of_plot):
    Bagging = BaggingClassifier()
    Bagging.fit(X_train,y_train)
    y_pred = Bagging.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    acc = (accuracy_score(y_test,y_pred))
    Y_of_plot.append(acc)

def BaggedTrees(X_train,y_train,Y_of_plot):
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 100
    BaggedTrees= BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(BaggedTrees, X_train, y_train, cv=kfold)
    BaggedTrees.fit(X_train,y_train)
    y_pred = BaggedTrees.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    Y_of_plot.append(results.mean())
    

def RandomForest(X_train,y_train,Y_of_plot):
    num_trees = 100
    max_features = 3
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    RandomForest = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(RandomForest, X_train, y_train, cv=kfold)
    RandomForest.fit(X_train,y_train)
    y_pred = RandomForest.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    Y_of_plot.append(results.mean())
    

# def LineerDiscriminant(X_train,y_train,Y_of_plot):
#     LineerDiscriminant = LinearDiscriminantAnalysis()
#     LineerDiscriminant.fit(X_train,y_train)
#     y_pred = LineerDiscriminant.predict(X_test)
#     print(y_pred)
#     print(y_test)
#     cm = confusion_matrix(y_test,y_pred)
#     print(cm)
#     acc = (accuracy_score(y_test,y_pred))
#     Y_of_plot.append(acc)

# def LineerDiscriminant(X_train,y_train,Y_of_plot):
#     LineerDiscriminant = LinearDiscriminantAnalysis()
#     LineerDiscriminant.fit(X_train,y_train)
#     y_pred = LineerDiscriminant.predict(X_test)
#     print(y_pred)
#     print(y_test)
#     cm = confusion_matrix(y_test,y_pred)
#     print(cm)
#     acc = (accuracy_score(y_test,y_pred))
#     Y_of_plot.append(acc)


#Some definitions
X_of_plot = ["FineTree","MediumTree","CoarseTree","LineerDiscriminant","QuadraticDiscriminant","LogisticRegressionM","GaussianNaiveBayes","LinearSVM","QuadricSVM","CubicSVM","GaussianSVM","FineKNN","MediumKNN","CubicKNN","WeightedKNN","XGBoost","AdaBoost","Bagging","BaggedTrees","RandomForest"]
Y_of_plot = []
seed = 0
number_of_features = 5

#Preprocessing
df = pd.read_csv('Difüzyon.csv')

x = df.iloc[:,23:130].values #Your Features
y = df.iloc[:,130:].values #Your Target


sc = MinMaxScaler()

X = sc.fit_transform(x)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

for i in range(6):
    Y_of_plot = []
    print(i)
    if i==0:
        X_new = SelectKBest(chi2, k=number_of_features).fit_transform(X_train, y_train)
        ffs = SelectKBest(chi2, k=number_of_features)
        fit = ffs.fit(X_train,y_train)   
    elif i==1:
        X_new = SelectKBest(f_classif, k=number_of_features).fit_transform(X_train, y_train)
        ffs = SelectKBest(f_classif, k=number_of_features)
        ffs.fit(X_train,y_train)
    elif i==2:
        X_new = SelectKBest(mutual_info_classif, k=number_of_features).fit_transform(X_train, y_train)
        ffs = SelectKBest(mutual_info_classif, k=number_of_features)
        ffs.fit(X_train,y_train)
    # elif i==3:
    #     m = RFECV(RandomForestClassifier(), scoring='accuracy')
    #     m = m.fit(X_train, y_train)
    #     m.score(X_train, y_train)
    #     X_new = m.fit_transform(X_train, y_train)
    #     #number_of_features = X_new.shape[1]
    # elif i==4:
    #     lsvc = LinearSVC().fit(X_train, y_train)
    #     model = SelectFromModel(lsvc, prefit=True)
    #     X_new = model.transform(X_train)
    #     #number_of_features = X_new.shape[1]
    # elif i==5:
    #     clf = ExtraTreesClassifier(n_estimators=50)
    #     clf = clf.fit(X_train, y_train)
    #     model = SelectFromModel(clf, prefit=True)
    #     X_new = model.transform(X_train)
    #     #number_of_features = X_new.shape[1]
        
        
            
    #Calling Machine Learning Models
    FineTree(X_train,y_train,Y_of_plot)
    MediumTree(X_train,y_train,Y_of_plot)
    CoarseTree(X_train,y_train,Y_of_plot)
    LineerDiscriminant(X_train,y_train,Y_of_plot)
    QuadraticDiscriminant(X_train,y_train,Y_of_plot)
    LogisticRegressionM(X_train,y_train,Y_of_plot)
    GaussianNaiveBayes(X_train,y_train,Y_of_plot)
    LinearSVM(X_train,y_train,Y_of_plot)
    QuadricSVM(X_train,y_train,Y_of_plot)
    CubicSVM(X_train,y_train,Y_of_plot)
    GaussianSVM(X_train,y_train,Y_of_plot)
    FineKNN(X_train,y_train,Y_of_plot)
    MediumKNN(X_train,y_train,Y_of_plot)
    # samples must be over 100
    #CoarseKNN(X_train,y_train,Y_of_plot)
    CubicKNN(X_train,y_train,Y_of_plot)
    WeightedKNN(X_train,y_train,Y_of_plot)
    XGBoost(X_train,y_train,Y_of_plot)
    AdaBoost(X_train,y_train,Y_of_plot)
    Bagging(X_train,y_train,Y_of_plot)
    BaggedTrees(X_train,y_train,Y_of_plot)
    RandomForest(X_train,y_train,Y_of_plot)
    # LineerDiscriminant(X_train,y_train,Y_of_plot)
    # LineerDiscriminant(X_train,y_train,Y_of_plot)
    
    # Plot Accuracy
    plt.xlabel("Model Names")
    plt.ylabel("Accuracy")
    plt.plot(X_of_plot,Y_of_plot)
    plt.show()

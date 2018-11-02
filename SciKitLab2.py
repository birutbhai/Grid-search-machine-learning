# This example has been taken from SciKit documentation and has been
# modifified to suit this project.


from __future__ import print_function


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import pandas as pd
from sklearn.preprocessing import StandardScaler

print(__doc__)

def read_data():
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None, names=['Class', 'Alcohol', 'malicAcid', 'Ash', 'alcalinityOfAsh', 'Magnesium', 'totalPhenols', 'flavanoids', 'nonflavanoidPhenols', 'Proanthocyanins', 'colorIntensity', 'Hue', 'OD280/OD315', 'Proline'])
    return df

def preprocess_and_split_data(df):
    temp = df.values
    X = temp[:,1:14]
    y = temp[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
    
    


def classification_report_gen(model_name, callback, tuned_parameters, X_train_param, X_test_param, y_train_param, y_test_param):
    print()
    scores = ['accuracy']
    print("Model being used: " + model_name)
    print()
    X_train = X_train_param
    X_test = X_test_param
    y_train = y_train_param
    y_test = y_test_param
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(callback(), tuned_parameters, cv=5, scoring='%s' % score)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print("Detailed confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("Accuracy Score: \n")
        print(accuracy_score(y_true, y_pred))
        return classification_report(y_true, y_pred)
    
if __name__ == "__main__":
    df = read_data()
    X_train, X_test, y_train, y_test = preprocess_and_split_data(df)
    models = { 

                  "Decision Tree":DecisionTreeClassifier,
                  "Neural Network":MLPClassifier,
                  "SVM":SVC,
                  "Gaussian Naive Bayes":GaussianNB,
                  "Logistic Regression":LogisticRegression,
                  "k-Nearest Neighbors":KNeighborsClassifier,
                  "Bagging":BaggingClassifier,
                  "Random Forest":RandomForestClassifier,
                  "AdaBoost Classifier":AdaBoostClassifier,
                  "Gradient Boosting Classfier":GradientBoostingClassifier,
                  "XGBoost":XGBClassifier               
              }
    tuned_parameters_list = [
                               [
                                    {#Decision Tree
                                            'max_depth':[1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]
                                            ,'min_samples_split':[2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 180, 240, 500, 1000, 2000]
                                            ,'min_weight_fraction_leaf':[0, 0.1, 0.2, 0.5,.001,.002,.004,.005,.0001]
                                            ,'max_features':[1, 3, 5, 7, 9, 11, 13]
                                    }
                               ],
                               [
                                    {#Neural Neteork
                                             'hidden_layer_sizes':[5, 10, 50, 100, 500, 1000, 10000]
                                             ,'activation':['identity', 'logistic', 'tanh', 'relu']
                                             ,'learning_rate':['constant', 'invscaling', 'adaptive']
                                             ,'max_iter':[5000, 7500, 10000, 50000]
                                             ,'early_stopping':[True]
                                    }
                               ],
                               [
                                    {#SVM
                                            'kernel': ['rbf', 'linear']
                                            ,'gamma': [1e-3, 1e-4, 1e-2, 1e-5, 1e-6, 1e-7, 1e-1]
                                            ,'C': [1, 10, 100, 1000]
                                            ,'max_iter': [1000, 2000,4000,5000,10000]
                                    }
                               ],
                               [
                                    {#Gaussian Naive Bayes
                                            'priors': [[ 0,0,1], [ 0.8, 0.1, 0.1], [0.33,0.33, 0.34], [ 0.5,  0.5, 0], [ 0, 0.3,  0.7], [0.6,0.2,0.2]]
                                    }
                               ],
                               [
                                    {#Logistic Regression
                                            'penalty': ['l1', 'l2']
                                            ,'tol': [.0001, .001, .005, .0005, .01, .05, .0009, .00001, .0008, .0003]
                                            ,'C': [1, 10, 100, 1000]
                                            ,'fit_intercept': [True, False]
                                    }
                               ],

                               [
                                    {#k-Nearest Neighbors
                                           'n_neighbors':[1,2,3,4,5,8,10,15,20,25,50]
                                           ,'weights':['uniform', 'distance']
                                           ,'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                                           ,'p':[1,2]
                                    }
                               ],
                               [
                                    {#Bagging
                                           'n_estimators':[5,10,15,20,30,40,50,70,75,80,100]
                                           ,'max_samples':[1, 3, 5, 7, 9, 11, 13]
                                           ,'max_features':[1, 3, 5, 7, 9, 11, 13]
                                           ,'random_state':[0,1,2]
                                    }
                               ],
                               [
                                    {#Random Forest
                                           'n_estimators':[5,10,15,20,30,40,50,70,75,80,100]
                                           ,'criterion':['gini', 'entropy']
                                           ,'max_depth':[10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]
                                           ,'max_features':[1, 3, 5, 7, 9, 11, 13]
                                    }
                               ],
                               [
                                    {#AdaBoost Classifier
                                           'n_estimators':[5,10,15,20,30,40,50,70,75,80,100]
                                           ,'learning_rate':[0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1]
                                           ,'algorithm':['SAMME', 'SAMME.R']
                                           ,'random_state':[0,1,2]
                                    }
                               ],
                               [
                                    {#Gradient Boosting Classfier
                                           'learning_rate':[0.1 ,0.3, 0.7, 0.9, 1]
                                           ,'n_estimators':[5,  20, 50, 100, 200]
                                           ,'max_depth':[10, 50, 80, 100, 150, 200, 300]
                                           ,'max_features':[1, 3, 7, 9, 11, 13]
                                    }
                               ],
                               [
                                    {#XGBClassifier
                                           'learning_rate':[0.1,0.4,0.6,0.8,1]
                                           ,'n_estimators':[5,15,30,50,75,80,100,200]
                                           ,'min_child_weight':[0,1,2,4,10,20,100]
                                           ,'booster':['gbtree','gblinear','dart']
                                    }
                               ]
    
                        ]
    i = 0
    warnings.simplefilter("ignore")
    for model_name, callback in models.items():
        s = classification_report_gen(model_name, callback, tuned_parameters_list[i], X_train, X_test, y_train, y_test)
        i = i+1
    

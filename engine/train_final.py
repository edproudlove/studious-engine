from sklearn import ensemble
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

import config

#I am going to stop letting the passenger ID being included as it seems to be realied 
#but i do not understan why.

def run_final():
    df = pd.read_csv(config.TRAINING_FILE_TO_BE_FOLDED)

    x_train = df.drop(['Transported'], axis=1).values
    y_train = df.Transported.values

    x_test = pd.read_csv(config.TEST_FILE)
    x_test = x_test.drop(['PassengerId'], axis=1)

    #clf = SVC(class_weight=None, gamma='auto', C=1)
    #clf = ensemble.RandomForestClassifier(criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 200)
    clf = XGBClassifier(
        alpha= 0, eta = 0.01, gamma = 0.1, max_depth = 10, min_child_weight= 7 
        )

    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)

    #converting into True and False
    preds_tf = [True if i == 1 else False for i in preds]
   
    
    df_for_csv = pd.read_csv(config.TEST_FILE)
    prediction_csv = pd.DataFrame(df_for_csv['PassengerId'])
    prediction_csv['Transported'] = preds_tf
    prediction_csv.to_csv('/Users/ethan/Desktop/Ethan/Python/ML/framework/output/predictions_space.csv', index=False)       


if __name__ == '__main__':
    run_final()


#This script got me an accuracy of 0.79425

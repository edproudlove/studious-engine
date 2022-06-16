from sklearn import ensemble
import argparse

import pandas as pd
import numpy as np

import config

#I am going to stop letting the passenger ID being included as it seems to be realied 
#but i do not understan why.

def run_final():
    df = pd.read_csv(config.TRAINING_FILE_TO_BE_FOLDED)

    x_train = df.drop(['Survived', 'PassengerId'], axis=1).values
    y_train = df.Survived.values

    x_test = pd.read_csv(config.TEST_FILE)
    x_test = x_test.drop(['PassengerId'], axis=1)

    clf = ensemble.RandomForestClassifier(n_estimators= 200, max_depth=8, criterion='gini')
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    
    df_for_csv = pd.read_csv(config.TEST_FILE)
    prediction_csv = pd.DataFrame(df_for_csv['PassengerId'])
    prediction_csv['Survived'] = preds
    prediction_csv.to_csv('/Users/ethan/Desktop/Ethan/Python/ML/framework/output/predictions.csv', index=False)       


if __name__ == '__main__':
    run_final()


#This script got me an accuracy of 0.79425

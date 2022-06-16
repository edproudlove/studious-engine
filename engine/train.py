import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher




def run(fold, model):
    #read training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    #training data is all the folds except the fold that is given 
    #also reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #validiation data for a given fold is one where the fold is equall to kfold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #drop the target column from the dataframe and convert it to a numpy array
    x_train = df_train.drop(['Transported'], axis=1).values
    y_train = df_train.Transported.values

    #same for validataion:
    x_valid = df_valid.drop(['Transported'], axis=1).values
    y_valid = df_valid.Transported.values

    #the model is imported:
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    #create predictions and print the evaluations
    preds = clf.predict(x_valid)

    accuracy = metrics.f1_score(y_valid, preds)
    print(f'Fold = {fold}, Accuracy = {accuracy}')

    #save the model.






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fold',
        type=int,
    )
    parser.add_argument(
        '--model',
        type=str,
    )

    args = parser.parse_args()

    run(
        fold = args.fold,
        model = args.model,
    )


    #python train.py --fold 0 --model rf
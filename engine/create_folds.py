import pandas as pd
import config

from sklearn import model_selection

#note -  this is for notamel kfolds not strainifed kfolds
#theese are for a data that is distributed equally
#stratified is on pg 24


if __name__ == '__main__':
    #get the original trainning data
    df = pd.read_csv(config.TRAINING_FILE_TO_BE_FOLDED)

    df['kfold'] = -1

    #get random data and put it into the folds
    df = df.sample(frac = 1).reset_index(drop=True)

    #fetch targets (in this case survived)
    y = df.Survived.values

    #get the kfolds from the model selection module
    kf = model_selection.StratifiedKFold(n_splits=10)

    #fill in the kfold col
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold
    
    #save for new csv:
    df.to_csv('/Users/ethan/Desktop/Ethan/Python/ML/framework/input/titanic/titanic_folds.csv', index=False)



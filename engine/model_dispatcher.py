from sklearn import tree
from sklearn import ensemble
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




models = {
    'decision_tree_gini': tree.DecisionTreeClassifier(
        criterion='gini'
    ),
    'decision_tree_entropy': tree.DecisionTreeClassifier(
        criterion='entropy'
    ),
    'rf': ensemble.RandomForestClassifier(),
    'xgb': XGBClassifier(),
    'svc': SVC(),
    'k_nearest': KNeighborsClassifier()

    



}

## list of commands:
#  python train.py --fold 0 --model rf
#  python train.py --fold 0 --model decision_tree_gini
#  python train.py --fold 0 --model decision_tree_entropy
#  python train.py --fold 0 --model xgb
#  python train.py --fold 0 --model svc
#  python train.py --fold 0 --model k_nearest

#rf and xgboost are the best atm.

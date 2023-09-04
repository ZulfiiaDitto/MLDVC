from imblearn.over_sampling import SMOTE
import pandas as pd
import yaml
import pickle 
from sklearn.ensemble import GradientBoostingClassifier
# you need to be in dvc ml project directory

def train():
    with open("./params.yaml", "r") as config:
        params = yaml.safe_load(config)["train"]
    
    train = pd.read_csv(params['dataset_csv'])

    numerical = [i for i in train.select_dtypes(include=['int64', 'float64']).columns if i not in ['Outcome Type']]

    X_train = train[numerical]
    y_train = train['Outcome Type']

    #sintetically balancing data
    smt = SMOTE()
    X_train, y_train = smt.fit_resample(X_train, y_train)

    # using clasifier
   # knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    knn = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['lr'],
                                     max_depth=params['max_depth'], random_state=params['random_seed'])
    knn.fit(X_train, y_train)

    with open('src/model/train.pkl', "wb") as fd:
        pickle.dump(knn, fd)

if __name__ == '__main__':
    train()
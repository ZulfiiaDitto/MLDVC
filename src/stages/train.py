from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import yaml
import pickle 
import argparse
# you need to be in scr directory

def train():
    with open("./params.yaml", "r") as config:
        params = yaml.safe_load(config)["train"]
        neibor = yaml.safe_load(config)["estimator"]['n_neighbors']

    print(params['dataset_csv'])
    train = pd.read_csv(params['dataset_csv'])

    numerical = [i for i in train.select_dtypes(include=['int64', 'float64']).columns if i not in ['Outcome Type']]

    X_train = train[numerical]
    y_train = train['Outcome Type']


    #sintetically balancing data
    smt = SMOTE()
    X_train, y_train = smt.fit_resample(X_train, y_train)

    # using clasifier
    knn = KNeighborsClassifier(n_neighbors=neibor)
    knn.fit(X_train, y_train)

    with open('model/train.pkl', "wb") as fd:
        pickle.dump(knn, fd)

if __name__ == '__main__':
    # args_parser = argparse.ArgumentParse()
    # args_parser.add_argument('--config', )
    train()
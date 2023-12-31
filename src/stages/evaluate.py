from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
import yaml
import pickle 
import json
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
#TODO: 
# 1. figurate out how to do graphs 
# 2. figurate out how to run with the parameters chnaged as we go 
# 3. how to connect to DVC studio


def evaluate():
    with open("params.yaml", "r") as config:
        params = yaml.safe_load(config)["test"]
        model = params['model']


    test = pd.read_csv(params['dataset_csv'])

    numerical = [i for i in test.select_dtypes(include=['int64', 'float64']).columns if i not in ['Outcome Type']]

    X_test = test[numerical]
    y_test = test['Outcome Type']

    with open(model, 'rb') as models_params:
        loaded_model = pickle.load(models_params)

    test_predictions = loaded_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.viridis)
    plt.tight_layout()
    plt.savefig("src/evaluation_artefacts/conf.png", pad_inches=5)


    with open('src/evaluation_artefacts/test.json', 'w') as fd:
        json.dump({'accuracy' : accuracy_score(y_test, test_predictions)}, fd)

    print('Model performance on the test set:')
    print(confusion_matrix(y_test, test_predictions))
    print(classification_report(y_test, test_predictions))
    print("Test accuracy:", accuracy_score(y_test, test_predictions))

if __name__ == '__main__':
    evaluate()
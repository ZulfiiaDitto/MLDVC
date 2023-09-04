from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd
import yaml
import pickle 
import json
from sklearn.neighbors import KNeighborsClassifier

from dvclive import Live
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
    cm = confusion_matrix(y_test, test_predictions, normalize='all')
    # cmd = ConfusionMatrixDisplay(cm, display_labels=['adapt','no'])
    # cmd.plot()

    with open('src/evaluation_artefacts/test.json', 'w') as fd:
        json.dump({'accuracy' : accuracy_score(y_test, test_predictions)}, fd)


    Live.log_plot("src/evaluation_artefacts/confusion matrix", cm, x="predicted", y="True",
    template="bar_horizontal", title="confusion matrix")
    
    print('Model performance on the test set:')
    print(confusion_matrix(y_test, test_predictions))
    print(classification_report(y_test, test_predictions))
    print("Test accuracy:", accuracy_score(y_test, test_predictions))

if __name__ == '__main__':
    evaluate()
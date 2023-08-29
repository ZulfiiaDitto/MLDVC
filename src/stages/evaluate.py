from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd
import yaml
import pickle 
from sklearn.neighbors import KNeighborsClassifier

with open("params.yaml", "r") as config:
    params = yaml.safe_load(config)["test"]
    neibor = yaml.safe_load(config)["estimator"]['n_neighbors']
    model = yaml.safe_load(config)["test"]['model']


test = pd.read_csv(params['dataset_csv'])

numerical = [i for i in test.select_dtypes(include=['int64', 'float64']).columns if i not in ['Outcome Type']]

X_test = test[numerical]
y_test = test['Outcome Type']

with open(model, 'rb') as models_params:
    loaded_model = pickle.load(models_params)

test_predictions = loaded_model.predict(X_test)


# with open('evaluation_artefacts/test.pkl', 'wb') as fd:
#     pickle.dump(confusion_matrix(y_test, test_predictions))
#     pickle.dump(classification_report(y_test, test_predictions))
#     pickle.dump(accuracy_score(y_test, test_predictions))


print('Model performance on the test set:')
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd
import yaml
import pickle 
from sklearn.neighbors import KNeighborsClassifier

params = yaml.safe_load(open("params.yaml"))["test"]
neibor = yaml.safe_load(open("params.yaml"))["estimator"]['n_neighbors']
model = yaml.safe_load(open("params.yaml"))["test"]['model']

test = pd.read_csv(params['dataset_csv'])

numerical = [i for i in test.select_dtypes(include=['int64', 'float64']).columns if i not in ['Outcome Type']]

X_test = test[numerical]
y_test = test['Outcome Type']

loaded_model = pickle.load(open(model, 'rb'))
test_predictions = loaded_model.predict(X_test)


# with open('evaluation_artefacts/test.pkl', 'wb') as fd:
#     pickle.dump(confusion_matrix(y_test, test_predictions))
#     pickle.dump(classification_report(y_test, test_predictions))
#     pickle.dump(accuracy_score(y_test, test_predictions))


print('Model performance on the test set:')
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))
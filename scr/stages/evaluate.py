from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd
import yaml
import pickle 

params = yaml.safe_load(open("params.yaml"))["test"]

test = pd.read_csv(params['dataset_csv'])

numerical = [i for i in test.select_dtypes(include=['int64', 'float64']).columns if i not in ['Outcome Type']]
# Get test data to test the classifier
X_test = test[numerical]
y_test = test['Outcome Type']

# Use the fitted model to make predictions on the test dataset
# Test data going through the Pipeline it's first imputed (with means from the train), scaled (with the min/max from the train data), and finally used to make predictions
test_predictions = classifier.predict(X_test)

print('Model performance on the test set:')
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 

def featurize():
    filename= yaml.safe_load(open("params.yaml"))["feature"]['dataset_csv']
    test_size = yaml.safe_load(open("params.yaml"))["feature"]['test_size']
    final = pd.read_csv(filename)
    #encoding the columns
    label_encoder = preprocessing.LabelEncoder()
    final['Sex upon Intake'] = label_encoder.fit_transform(final['Sex upon Intake'])
    final['Sex upon Outcome'] = label_encoder.fit_transform(final['Sex upon Outcome'])

    label_enc_breed = preprocessing.LabelEncoder()
    final['Breed'] = label_enc_breed.fit_transform(final['Breed'])

    label_enc_color = preprocessing.LabelEncoder()
    final['Color'] = label_enc_color.fit_transform(final['Color'])

    label_enc_pet_type = preprocessing.LabelEncoder()
    final['Pet Type'] = label_enc_pet_type.fit_transform(final['Pet Type'])
    label_enc_intake_type = preprocessing.LabelEncoder()
    final['Intake Type'] = label_enc_intake_type.fit_transform(final['Intake Type'])

    label_enc_intake_condition = preprocessing.LabelEncoder()
    final['Intake Condition'] = label_enc_intake_condition.fit_transform(final['Intake Condition'])

    train, test = train_test_split(final, test_size=test_size, shuffle=True)

    test.to_csv('src/data/splits/test.csv')
    train.to_csv('src/data/splits/train.csv')

if __name__ =='__main__':
    featurize()
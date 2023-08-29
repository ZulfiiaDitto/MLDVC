import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from data_load import data_load 

test_size = yaml.safe_load(open("params.yaml"))["data_split"]['test_size']

final = data_load()

#import label encoder
from sklearn import preprocessing 
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

test.to_csv('data/splits/test.csv')
train.to_csv('data/splits/train.csv')
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
# you need to be in scr directory

params = yaml.safe_load(open("params.yaml"))["data_load"]

def get_data():
  """function pulls data from url, modify if needed. 
    Data is already saved in folder data"""
  final = pd.read_csv('https://raw.githubusercontent.com/aws-samples/aws-machine-learning-university-accelerated-tab/master/data/review/review_dataset.csv')
  final.drop('Name',axis = 1, inplace=True)
  final.fillna('Missing', inplace = True)
  final['difference'] = final['Age upon Outcome Days'] - final['Age upon Intake Days']
  final.to_csv('data_shelter.csv')

def data_load():
  """Loading data from the file"""
  df = pd.read_csv(params["dataset_csv"])
  return df


test_size = yaml.safe_load(open("params.yaml"))["data_split"]['test_size']

df = data_load()

train, test = train_test_split(df, test_size=test_size, shuffle=True)

test.to_csv('data/splits/test.csv')
train.to_csv('data/splits/train.csv')

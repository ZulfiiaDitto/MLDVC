import pandas as pd
import yaml


# this script not need to be run -> since all data had been loaded already

with open("params.yaml", "r") as config:
  params = yaml.safe_load(config)["data_load"]

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
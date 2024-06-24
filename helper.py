import joblib
import numpy as np
import pandas as pd

encoder1 = joblib.load(r'Model\author_encoder.joblib')
encoder2 = joblib.load(r'Model\geometry_encoder.joblib')

def author_encoder(input_value):
    return encoder1.transform([input_value])[0]

def geometry_encoder(input_value):
  return encoder2.transform([input_value])[0]

def get_prediction(data, model):
  val = model.predict(data)
  return val[0]
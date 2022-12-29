import pandas as pd
import joblib 

test=[[3,2,2,1,5]]

model=joblib.load('model.pkl')

print(model.predict(test))
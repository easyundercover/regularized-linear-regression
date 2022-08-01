#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

#Load data
url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv"
df_raw = pd.read_csv(url)

#Copy df into a new one in order to make transformations
df = df_raw.copy()

#Transforming categorical variables
df.STATE_FIPS = pd.Categorical(df.STATE_FIPS)
df.CNTY_FIPS = pd.Categorical(df.CNTY_FIPS)
df.Urban_rural_code = pd.Categorical(df.Urban_rural_code)

#Splitting data to avoid bias
X= df.drop(['CNTY_FIPS','fips','Active Physicians per 100000 Population 2018 (AAMC)','Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)','Active General Surgeons per 100000 Population 2018 (AAMC)','Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)','Total physician assistants (2019)','Total physician assistants (2019)','Total Hospitals (2019)','Internal Medicine Primary Care (2019)','Family Medicine/General Practice Primary Care (2019)','STATE_NAME','COUNTY_NAME','ICU Beds_x','Total Specialist Physicians (2019)'], axis=1)
y = df['ICU Beds_x']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)

#LASSO | Default alpha 10%
pipeline = make_pipeline(StandardScaler(), Lasso(alpha=10)) #Pipeline to normalize data and then apply Lasso
pipeline.fit(X_train, y_train)
print(pipeline[1].coef_, pipeline[1].intercept_)

coef_list=pipeline[1].coef_
loc=[i for i, e in enumerate(coef_list) if e != 0]
col_name=df.columns
col_name[loc]

# Variable selection
list_final=[]
list_final.extend(loc)
my_IV=list(set(list_final))
col_name[my_IV]
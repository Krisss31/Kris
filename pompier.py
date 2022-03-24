import pandas as pd
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title("pompier")

def file_selector():
   file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
   if file is not None:
      data = pd.read_csv(file)
      return data
   else:
      st.text("Please upload a csv file")   

df = file_selector()

st.write(df)

def set_features(self):
   self.features = st.multiselect("Please choose the features including target variable that go into the model", self.columns )
   
df.feature = set_features(df)
st.write(df.feature)

def prepare_data(self, split_data, train_test):
   # Reduce data size
   data = self.data[self.features]
   data = data.sample(frac = round(split_data/100,2))

prepare_data(df, split_data, train_test)

   # Impute nans with mean for numeris and most frequent for categoricals
cat_imp = SimpleImputer(strategy="most_frequent")
if len(data.loc[:,data.dtypes == 'object'].columns) != 0:
   data.loc[:,data.dtypes == 'object'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'object'])
   imp = SimpleImputer(missing_values = np.nan, strategy="mean")
   data.loc[:,data.dtypes != 'object'] = imp.fit_transform(data.loc[:,data.dtypes != 'object'])

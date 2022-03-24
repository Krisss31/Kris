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
      
if st.sidebar.checkbox("Display data", False):
    st.subheader("Show Mushroom dataset")
    st.write(df)

# Set features
features_options = df.columns
features = st.multiselect("Please choose the features including target variable that go into the model", features_options)
df = df[features]

# Set target column
target_options = df.columns
target = st.sidebar.selectbox("Please choose target column", (target_options))

st.write("la target est:", target)

@st.cache(persist=True)
def split(df):
   y = df['target']
   x = df.drop(columns=["target"])
   x_train, x_test, y_train, y_test =     train_test_split(x,y,test_size=0.3, random_state=0)
   return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split(df)

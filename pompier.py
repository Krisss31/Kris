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

data = file_selector(file)
      
def set_features():
   self.features = st.multiselect("Please choose the features including target variable that go into the model", self.data.columns)

self.features = set_features()


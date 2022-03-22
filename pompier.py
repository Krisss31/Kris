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



def file_selector(self):
   file = st.sidebar.file_uploader("Choose a pikle file", type="csv")
   if file is not None:
      df=pd.read_pickle(file)
      return df
   else:
      st.text("Please upload a pikle file")

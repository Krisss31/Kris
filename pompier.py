import pandas as pd
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, classification_report, f1_score

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
    st.write(df)

# Set features
#if df is not None:
   #features_options = df.columns
   #features = st.multiselect("Please choose the features including target variable that go into the model", features_options)
   #df = df[features]
   
choice_features = st.sidebar.multiselect("Do you want to choice the feature", ("yes", "no"))
if choice_features == "yes":
   if df is not None:
      features_options = df.columns
      features = st.multiselect("Please choose the features including target variable that go into the model", features_options)
      df = df[features]
      st.write(df)

# Set target column
if df is not None:
   target_options = df.columns
   target = st.sidebar.selectbox("Please choose target column", (target_options))
      
@st.cache(persist=True)
#@st.cache(suppress_st_warning=True)
def split(df):
      y = df[target]
      x = df.drop(columns=[target])
      x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
      return x_train, x_test, y_train, y_test

if df is not None:
   x_train, x_test, y_train, y_test = split(df)

   
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression", "Random Forest", "DecisionTreeClassifier"))

def metrics(metrics_list):
   if "Confusion Matrix" in metrics_list:
      st.subheader("Confusion Matrix")
      plot_confusion_matrix(model, x_test, y_test, display_labels = [1, 2, 3])
      st.pyplot()

   if "Classification Report" in metrics_list:
      st.subheader("Classification Report")
      return st.text(classification_report(y_test, y_pred))
 
if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metric = st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix", "Classification Report"))
    st.subheader("Logistic Regression Results")
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)      
    st.write("Accuracy: ", accuracy)
    #st.write("Precision: ", precision_score(y_test, y_pred))
    #st.write("Recall: ", recall_score(y_test, y_pred))
    metrics(metric)

if classifier == "Random Forest":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of trees in the forest",10, 500, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 50, step =1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    
    metrics = st.sidebar.multiselect("Do you want Confusion Metrix", ("yes", "no"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= bootstrap, n_jobs=-1 )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        #st.write("Precision: ", precision_score(y_test, y_pred,labels=class_names).round(2))
        #st.write("Recall: ", recall_score(y_test, y_pred,labels=class_names).round(2))
        plot_metrics(metrics)
         
if classifier == "DecisionTreeClassifier":
    st.sidebar.subheader("Hyperparameters")
    criterion= st.sidebar.multiselect("Criterion",("entropy", "giny"))
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 50, step =1, key="max_depth")
    
    metrics = st.sidebar.multiselect("Do you want Confusion Metrix", ("yes", "no"))
    
      
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("DecisionTreeClassifier")
        model = DecisionTreeClassifier(criterion = criterion,  max_depth = max_depth)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

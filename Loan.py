import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer, precision_score, precision_recall_curve
from sklearn.metrics import  roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


st.set_option("deprecation.showPyplotGlobalUse", False)
def main():
  st.title(" Application de machine learning pour le risque de crédit")
  st.subheader("Auteur : Franck Théotiste")
  
# Fonction d'importation des données

def load_data():
  data = pd.read_csv('Loan_data.csv')  
  return data
# Affichage de la table des données
df = load_data()
df_sample = df.sample(100)
if st.sidebar.checkbox("Afficher les données brutes", False):
   st.subheader("Jeu de données 'Risque de crédit' : Echantillon  de 100 observations")
   st.write(df_sample)

y = df['default']
X = df.drop('default', axis=1)
#st.write("Shape of dataset", X.shape)
#st.write("number of classes", len(np.unique(y.shape)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 42)
 #  X_train, X_test, y_train, y_test = split(df)

classifier = st.sidebar.selectbox("Classificateur", ("DecisionTree", "Logistic Regression"))

def add_parameter_ui(classifier):
    params = dict()
    if classifier == "DecisionTree":
        max_depth = st.sidebar.slider("max_depth", 1, 3)
        params["max_depth"] = max_depth
    elif classifier == "Logistic Regression":  # Correction ici
        max_iter = st.sidebar.slider("max_iter", 1, 200)
        params["max_iter"] = max_iter
    return params

params = add_parameter_ui(classifier)

def get_classifier(classifier, params):
    if classifier == "DecisionTree":
        clf = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=1234)
    elif classifier == "Logistic Regression": 
        clf = LogisticRegression(max_iter=params["max_iter"], random_state=1234)
    return clf

clf = get_classifier(classifier, params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

                      

#Analyse de la performance des modèles 
def plot_perf(graphes):
     if 'confusion matrix' in graphes:
       st.subheader('Matrice de confusion')
       ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
     st.pyplot()
  
     if 'ROC curve' in graphes:
       st.subheader('Courbe ROC')
       RocCurveDisplay.from_estimator(model, X_test, y_test)
     st.pyplot()
  
     if 'Precision-Recall curve' in graphes:
       st.subheader('Courbe Precision-Recall')
       PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
       st.pyplot()
      
#Arbre de décision

Graph_perf_tree = st.sidebar.multiselect("Choisir un graphique de performance du modèle ML", ("confusion matrix", "ROC curve", "Precision-Recall curve"), key="tree_graph_perf") 
if st.sidebar.button("Exécution", key="classify"):
    st.subheader("DecisionTree Results")
    #Initialisation d'un objet DecisionTreeClassifier
    model=DecisionTreeClassifier(max_depth=params["max_depth"], random_state=42)
    #Entrainement de l'algorithme
    model.fit(X_train, y_train)
    #Prédictions
    y_pred = model.predict(X_test)
    #Métriques de performances
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    #Afficher les métriques dans l'application
    st.write("Accuracy :", round(accuracy, 3))
    st.write("Precision :", round(precision, 3))
    st.write("Recall :", round(recall, 3))
      
    #Afficher les graphiques de performances
    plot_perf(Graph_perf_tree)


#Régression logistique
     
Graph_perf_logreg = st.sidebar.multiselect("Choisir un graphique de performance du modèle ML", ("confusion matrix", "ROC curve", "Precision-Recall curve"), key="logreg_graph_perf") 
if st.sidebar.button("Exécution", key="classify1"):
    st.subheader("Logistic Regression Results")
    #Initialisation d'un objet LogisticRegression
    model=LogisticRegression(max_iter=params["max_iter"], random_state=42)
    #Entrainement de l'algorithme
    model.fit(X_train, y_train)
    #Prédictions
    y_pred = model.predict(X_test)
    #Métriques de performances
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    #Afficher les métriques dans l'application
    st.write("Accuracy :", round(accuracy, 3))
    st.write("Precision :", round(precision, 3))
    st.write("Recall :", round(recall, 3))
      
    #Afficher les graphiques de performances
    plot_perf(Graph_perf_logreg)

def plot_perf(graphes):
    if 'confusion matrix' in graphes:
        st.subheader('Matrice de confusion')
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        st.pyplot()
  
    if 'ROC curve' in graphes:
        st.subheader('Courbe ROC')
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        st.pyplot()
 
    if 'Precision-Recall curve' in graphes:
        st.subheader('Courbe Precision-Recall')
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
        st.pyplot()
      

if __name__ == '__main__':
   main()  
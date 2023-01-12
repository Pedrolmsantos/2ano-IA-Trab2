import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
def importdata():
    list = pd.read_csv('speedDating_trab.csv')
    list.head()
    list = list.dropna(how='any',axis=0) 
    return list

def main():
    data = importdata()
    clean_data = data.copy()
    X = data.copy()
    del X['match']
    y = clean_data[['match']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=101)
    gauss = GaussianNB().fit(X_train,y_train.values.ravel())
    y_predicted = gauss.predict(X_test)
    print("Percentagem de Acertos Gauss = ")
    print(accuracy_score(y_test,y_predicted)*100)
    print(confusion_matrix(y_test,y_predicted))
    clf1 = DecisionTreeClassifier(criterion = 'entropy')
    clf1.fit(X_train,y_train)
    y_predicted = clf1.predict(X_test)
    print("Percentagem de Acertos ID3 = ")
    print(accuracy_score(y_test,y_predicted)*100)
    print(confusion_matrix(y_test,y_predicted))
if __name__=="__main__":
    main()
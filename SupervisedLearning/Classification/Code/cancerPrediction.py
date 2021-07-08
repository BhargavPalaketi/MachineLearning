
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
 
# converting sklearn.dataset to a DataFrame
def conv_to_dataframe():
    

    data = pd.DataFrame(data = cancer['data'],columns = cancer['feature_names'])
    #data['target'] = cancer['target']
    data.insert(30,'target',cancer['target'],allow_duplicates=True)
    return data


# To know how many malignante(0) and how many begign(1) are their in dataset
def output_distribution():
    cancerdf = conv_to_dataframe()
    
    maligant = np.where(cancerdf['target'] == 0 )
    benign = np.where(cancerdf['target'] == 1 )
    
    data = [np.size(maligant) , np.size(benign)]
    res = pd.Series(data,index=['malignant', 'benign'])
    return res


def split_data_as_X_y():
    cancerdf = conv_to_dataframe()
    X = cancerdf[cancerdf.columns[[0:30]]
    y = cancerdf['target']  
    return X, y

from sklearn.model_selection import train_test_split

def split_data_to_train_test():
    X, y = split_data_as_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
    
    return X_train, X_test, y_train, y_test


from sklearn.neighbors import KNeighborsClassifier

def classification_model():
    X_train, X_test, y_train, y_test = split_data_to_train_test()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    KnnModel = knn.fit(X_train,y_train)
    return KnnModel 


def predict_on_mean_values():
    cancerdf = conv_to_dataframe()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    # predicting on the mean of every feature
    knn = classification_model()
    prediction = knn.predict(means)
    
    return prediction

def model_on_test_data():
    X_train, X_test, y_train, y_test = split_data_to_train_test()
    knn = classification_model()
    predict_test = knn.predict(X_test)
    
    return predict_test


def model_accuracy():
    X_train, X_test, y_train, y_test = split_data_to_train_test()
    knn = classification_model()
    
    acur = knn.score(X_test,y_test)
    return acur

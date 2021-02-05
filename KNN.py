#k-Nearest Neighbors Classification
from sklearn.neighbors import KNeighborsClassifier
import joblib

joblib_file = "KNN_model.pkl"
def KNN(x_train,x_test,y_train):
    #create model
    model = KNeighborsClassifier(n_neighbors=3)
    #fit model
    model.fit(x_train, y_train)
    #prediction test data
    y_pred = model.predict(x_test)
    #save model with joblib
    #joblib.dump(model, joblib_file)
    
    return y_pred

#Support Vector Machines Classification
from sklearn.svm import SVC
import joblib
joblib_file = "SVM_model.pkl"
def SVM(x_train,x_test,y_train):
    #create SVM model
    model = SVC()
    #fit model
    model.fit(x_train, y_train)
    #prediction test data
    y_pred = model.predict(x_test)
    #save model with joblib
    #joblib.dump(model, joblib_file)
    
    return y_pred

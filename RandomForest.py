#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
random.seed(54)
joblib_file = "RandomForest_model.pkl"

def randomForest(x_train,x_test,y_train):
    #create a random forest model
    
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    #fit model
    model.fit(x_train, y_train)
    #predict model
    y_pred=model.predict(x_test)

    #save model with joblib
    #joblib.dump(model, joblib_file)

           
    return y_pred

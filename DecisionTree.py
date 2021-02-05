#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
import joblib
import random
random.seed(54)
joblib_file = "DecisionTree_model.pkl"
def decisionTree(x_train,x_test,y_train):
    #create decision tree model
    model = DecisionTreeClassifier(random_state=42)
    #fit model
    model.fit(x_train, y_train)
    #prediction test data
    y_pred = model.predict(x_test)
    #save model with joblib
    #joblib.dump(model, joblib_file)
    return y_pred

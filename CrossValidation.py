from sklearn.model_selection import train_test_split


def CV(clean_messages,data):
   #split data train and test
   #x_train=train data,x_test=test data, y_train=train labels, y_test=test_labels
   #%20 test %80 train
   
   x_train, x_test, y_train, y_test = train_test_split(clean_messages, data['labels'], test_size=.2, random_state=1,stratify=data['labels'])

   return x_train, x_test, y_train, y_test

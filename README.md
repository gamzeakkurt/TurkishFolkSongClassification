# TurkishFolkSongClassification
Turkish Folk Song Text Classifier
### Preprocessing
The data is Turkish Folk Song that has Sivas, Erzurum, and, Erzincan regions.  The type of data is text and also, samples of each region are unbalanced. 
In **Figure 1.0**, **Sivas**, **Erzurum**, and **Erzincan** regions have **639**, **514**, **461** samples respectively.

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/114288639-7f751800-9a7a-11eb-9d52-22db8899cb1f.png" /></p>
<p align="center">
  <b>Figure 1.0</b>
</p>

To solve this problem, We used the resampling method. The purpose of this approach is to provide that each label has the same size. In the resampling method, there are 2 different approaches. These are under-sampling and over-sampling. Under-sampling removes samples from the majority class. Over-sampling adds more examples from minority class. Among these approaches, we preferred to use the over-sampling method. The reason why I did not choose the under-sampling method is that we do not have huge data, so when we remove samples from the dataset, important data can be deleted and we may encounter an under-fitting problem.
You can see the result in **Figure 1.1**. Each region has **639** samples.

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/114288850-60c35100-9a7b-11eb-853b-32e166ebbffe.png" /></p>
<p align="center">
  <b>Figure 1.1</b>
</p>

### Cleaning
After resampling, we cleaned the data in order to get rid of unnecessary information. In this process,  text values are converted from uppercase to lowercase, removed new lines, punctuation, digits, and special characters. For Turkish stop words, we removed these words from data using the **NLTK** library. You can see more information about it in the 'preprocessing.py' file.

### Feature Extraction

We preferred a **Bag of Words** (**BOW**) model that is one of the methods in natural language processing for feature extraction. In this model, the frequency of each word is calculated and it is used for training. BOW traverses all text in the data, figures out a set of words, and stores them as features.  In python, scikit-learn's **CountVectorizer** is a **BOW** implementation to convert a collection of text documents to a vector of term counts. In the project, we used this model with default settings and converted words to **NumPy** array.
 
 ### Cross Validation
 
 We split the data into the proportion of **80%** train and **20%** test. Also, each class sample is split with this rate. We used the **train_test_split** of the **scikit-learn** method for this task.
 
 ### Classifications
 
We used five different machine learning algorithms to train and test the dataset. These algorithms are  **Random Forest**, **Naive Bayes**, **Decision Tree**, **K-Nearest Neighbor**, and **Support Vector Machine**. 
The purpose of using different machine learning algorithms is to compare the results and measure the performance of the model. 
Naive Bayes has the highest accuracy when we compare other algorithms. The accuracy is **80.21**.

<p align="center"> <b> Table 1.0 </b></p>
<table border="1" cellspacing="0" cellpadding="1" align="center">
    <thead>
        <tr>
            <th align="left">Classifications</th>
            <th align="center">Accuracy</th>
            <th align="center">Precision</th>
            <th align="center">Recall</th>
            <th align="center">F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="left">Naive Bayes</td>
            <td align="center">80.21</td>
            <td align="center">80.45</td>
            <td align="center">80.20</td>
            <td align="center">80.45</td>
        </tr>
         <tr>
            <td align="left">Random Forest</td>
            <td align="center">77.86</td>
            <td align="center">77.43</td>
            <td align="center">77.86</td>
            <td align="center">77.43</td>
        </tr>
         <tr>
            <td align="left">SVM</td>
            <td align="center">76.56</td>
            <td align="center">76.41</td>
            <td align="center">76.56</td>
            <td align="center">76.41</td>
        </tr>
         <tr>
            <td align="left">Decision Tree</td>
            <td align="center">76.3</td>
            <td align="center">76.10</td>
            <td align="center">76.3</td>
            <td align="center">76.10</td>
        </tr>
         <tr>
            <td align="left">KNN</td>
            <td align="center">51.56</td>
            <td align="center">47.80</td>
            <td align="center">51.56</td>
            <td align="center">57.80</td>
        </tr>
    </tbody>
</table>
</p>

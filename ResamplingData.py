from sklearn.utils import resample
import pandas as pd

def Resample(data):
    label_0=data[data.labels==0] #erzincan
    label_1=data[data.labels==1] #erzurum 
    label_2=data[data.labels==2] #sivas

    # upsample minority
    label_0_upsampled = resample(label_0,
                              replace=True, # sample with replacement
                              n_samples=len(label_2), # match number in majority class
                              random_state=27) # reproducible results

    label_1_upsampled = resample(label_1,
                              replace=True, # sample with replacement
                              n_samples=len(label_2), # match number in majority class
                              random_state=27)
    # combine majority and upsampled minority
    upsampled = pd.concat([label_2, label_0_upsampled,label_1_upsampled])

    return upsampled
   

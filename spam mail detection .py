print("Making a ML moddel to detect Spam mail")
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

mail=pd.read_csv(r"D:\mail_data.csv")
print(mail.shape)
print(mail.head())
print(mail.info())
print(mail.describe())

le=LabelEncoder()
mail['siya']=le.fit_transform(mail['Category'])
maild=mail.drop(columns='Category' , axis=1)
print(maild.head())

X=mail['Message']
Y=mail['siya']

print(X.head())
print(Y.head())
ritik=maild['siya'].value_counts()
print(ritik)
# 0=4825 , 1=747
#applying somote to manage imbalanced data

tf=TfidfVectorizer(stop_words='english',lowercase=True)
Xtfidf=tf.fit_transform(X)

X_train , X_test , Y_train , Y_test=train_test_split(Xtfidf,Y, random_state=2 , test_size=0.2 , stratify=Y)
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=42)
x_trainsm , y_trainsm=sm.fit_resample(X_train , Y_train)

#training ml model 

model=LogisticRegression()

model.fit(x_trainsm , y_trainsm)

trainpredict=model.predict(x_trainsm)
print("The accuracy of train  data" , accuracy_score( trainpredict , y_trainsm))


testpredict=model.predict(X_test)
print("The accuracy score of test data" , accuracy_score(testpredict , Y_test))





#siya=mail.isnull().sum()
#print(siya)




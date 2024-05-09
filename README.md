# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ILAIYADEEPAN K
RegisterNumber:  212223230080
*/
```

```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Data Head
![image](https://github.com/ILAIYADEEPAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473334/5526194d-3c99-42e2-b1d7-7d13febd67e1)

## Data Info
![image](https://github.com/ILAIYADEEPAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473334/bd1a4912-bb15-4a67-a0e5-238765f30966)

## Data Isnull
![image](https://github.com/ILAIYADEEPAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473334/ad5f88fa-0ddd-492a-92c8-f471ec597862)

## Y_pred
![image](https://github.com/ILAIYADEEPAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473334/f6479898-4791-41cf-881c-5792c2d08d62)


## Accuracy

![image](https://github.com/ILAIYADEEPAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473334/616c09d9-98a0-4f9c-87f0-775195c965a6)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

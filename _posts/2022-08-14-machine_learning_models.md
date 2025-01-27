---
layout: single
title:  "[딥러닝 부트캠프] 1. Machine Learning Models"
---  
  
2022년 여름방학에 3주간 서울시립대에서 진행되었던 딥러닝 부트캠프 관련 코드이다.  
신경망을 적용해보기 이전 다양한 머신러닝 모델을 활용해 성능을 측정하였다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
%matplotlib inline

import timeit

rc('font', family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False
```


```python
df = pd.read_excel('data.xlsx')
```


```python
df.rename(columns=df.iloc[0], inplace=True)
```


```python
df = df.drop(df.index[0])
```


```python
df = df.dropna(subset=['label'])
```


```python
df = df.dropna(axis=1)
```


```python
df = df.drop(columns=['번호','날짜','시간','시편상태','요약','수막두께'])
```


```python
df = df.apply(pd.to_numeric)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 39007 entries, 1945 to 43193
    Data columns (total 36 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   외부 대기온도         39007 non-null  float64
     1   최대 대기온도         39007 non-null  float64
     2   최소 대기온도         39007 non-null  float64
     3   외부 상대습도         39007 non-null  int64  
     4   이슬점 온도          39007 non-null  float64
     5   풍속              39007 non-null  float64
     6   풍정              39007 non-null  float64
     7   최대풍속            39007 non-null  float64
     8   체감온도            39007 non-null  float64
     9   열지수             39007 non-null  float64
     10  THW             39007 non-null  float64
     11  THSW            39007 non-null  float64
     12  기압              39007 non-null  float64
     13  강우량             39007 non-null  float64
     14  강우강도            39007 non-null  float64
     15  Rad.            39007 non-null  int64  
     16  Energy          39007 non-null  float64
     17  Rad.            39007 non-null  int64  
     18  UV              39007 non-null  float64
     19  UV Dose         39007 non-null  float64
     20  Hi UV           39007 non-null  float64
     21  Head D-D        39007 non-null  float64
     22  Cool D-D        39007 non-null  float64
     23  In Temp         39007 non-null  float64
     24  In Hum          39007 non-null  int64  
     25  In Dew          39007 non-null  float64
     26  In Heat         39007 non-null  float64
     27  In EMC          39007 non-null  float64
     28  In Air Density  39007 non-null  float64
     29  ET              39007 non-null  float64
     30  노면 온도1          39007 non-null  float64
     31  Wind Samp       39007 non-null  int64  
     32  Wind Tx         39007 non-null  int64  
     33  ISS Recept      39007 non-null  float64
     34  Arc. Int.       39007 non-null  int64  
     35  label           39007 non-null  int64  
    dtypes: float64(28), int64(8)
    memory usage: 11.0 MB



```python
df.columns
```




    Index(['외부 대기온도', '최대 대기온도', '최소 대기온도', '외부 상대습도', '이슬점 온도', '풍속', '풍정',
           '최대풍속', '체감온도', '열지수', 'THW', 'THSW', '기압', '강우량', '강우강도', 'Rad.',
           'Energy', 'Rad. ', 'UV ', 'UV Dose', 'Hi UV', 'Head D-D', 'Cool D-D',
           'In Temp', 'In Hum', 'In Dew', 'In Heat', 'In EMC', 'In Air Density',
           'ET ', '노면 온도1', 'Wind Samp', 'Wind Tx', 'ISS Recept', 'Arc. Int.',
           'label'],
          dtype='object')




```python
sns.countplot(x='label',data=df)
```




    <AxesSubplot:xlabel='label', ylabel='count'>




    
![png](/assets/images/output_10_1.png)
    


# Logistic Regression


```python
X = df.drop('label',axis=1).values
y = df['label'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```


```python
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
```


```python
%%timeit

logmodel.fit(X_train, y_train)
```


    337 ms ± 18.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
predictions = logmodel.predict(X_test)
```


```python
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       0.89      0.94      0.91      3796
               1       0.96      0.92      0.94      5956
    
        accuracy                           0.93      9752
       macro avg       0.93      0.93      0.93      9752
    weighted avg       0.93      0.93      0.93      9752
    



```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
```




    array([[3582,  214],
           [ 454, 5502]])



# KNN


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
```


```python
scaler.fit(df.drop('label',axis=1))
```



```python
scaled_features = scaler.transform(df.drop('label', axis=1))
```


```python
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
```


```python
from sklearn.model_selection import train_test_split
X = df_feat
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier(n_neighbors=3)
```


```python
%%timeit

knn.fit(X_train, y_train)
```

    4.02 ms ± 560 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
predictions = knn.predict(X_test)
```



```python
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

    [[4553   41]
     [  75 7034]]
                  precision    recall  f1-score   support
    
               0       0.98      0.99      0.99      4594
               1       0.99      0.99      0.99      7109
    
        accuracy                           0.99     11703
       macro avg       0.99      0.99      0.99     11703
    weighted avg       0.99      0.99      0.99     11703
    


# Support Vector Machine


```python
from sklearn.model_selection import train_test_split
X = df.drop('label',axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
```


```python
from sklearn.svm import SVC
svc_model = SVC()
```


```python
%%timeit

svc_model.fit(X_train,y_train)
```

    8.43 s ± 65.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
```

    [[4485  245]
     [1043 5930]]
                  precision    recall  f1-score   support
    
               0       0.81      0.95      0.87      4730
               1       0.96      0.85      0.90      6973
    
        accuracy                           0.89     11703
       macro avg       0.89      0.90      0.89     11703
    weighted avg       0.90      0.89      0.89     11703
    


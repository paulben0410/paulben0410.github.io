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




    
![png](output_10_1.png)
    


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

    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


    337 ms ± 18.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



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




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>StandardScaler()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div>




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

    /opt/homebrew/Caskroom/miniforge/base/envs/jedi/lib/python3.8/site-packages/sklearn/neighbors/_classification.py:237: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)



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
    


---
title: Propensity Score Matching on AirAsia Insurance Upsell Data (Causal Inference)
tags: [Causal Inference, Propensity Score Matching, Propensity Modelling, Python, Kaggle]
style: fill
color: warning
description: Propensity Score Matching as a method to descriptive data science modelling exemplified with AirAsia bookings dataset from Kaggle.
---


```python
import os
print (os.environ['CONDA_DEFAULT_ENV'])

from platform import python_version

import multiprocessing
!python --version
import platform
print(platform.platform())
print("cpu cores: {0}".format(multiprocessing.cpu_count()))
```

    abtest
    Python 3.7.11
    Windows-10-10.0.19041-SP0
    cpu cores: 12
    


```python
# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression

sns.set_style('darkgrid')
```


```python
df = pd.read_csv('AncillaryScoring_insurance.csv',encoding='latin-1')
df = df.sample(n=8000).reset_index()
df['index'] = df.index

df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>PAXCOUNT</th>
      <th>SALESCHANNEL</th>
      <th>TRIPTYPEDESC</th>
      <th>PURCHASELEAD</th>
      <th>LENGTHOFSTAY</th>
      <th>flight_hour</th>
      <th>flight_day</th>
      <th>ROUTE</th>
      <th>geoNetwork_country</th>
      <th>BAGGAGE_CATEGORY</th>
      <th>SEAT_CATEGORY</th>
      <th>FNB_CATEGORY</th>
      <th>INS_FLAG</th>
      <th>flightDuration_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>27345</td>
      <td>1</td>
      <td>Internet</td>
      <td>RoundTrip</td>
      <td>47</td>
      <td>3</td>
      <td>4</td>
      <td>Sun</td>
      <td>DPSHND</td>
      <td>Indonesia</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7714</td>
      <td>1</td>
      <td>Internet</td>
      <td>RoundTrip</td>
      <td>2</td>
      <td>20</td>
      <td>19</td>
      <td>Fri</td>
      <td>DACMEL</td>
      <td>Australia</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8.83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>47603</td>
      <td>1</td>
      <td>Internet</td>
      <td>RoundTrip</td>
      <td>37</td>
      <td>6</td>
      <td>5</td>
      <td>Sun</td>
      <td>ICNSYD</td>
      <td>Australia</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8.58</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = df[['INS_FLAG']]
df = df.drop(columns = ['INS_FLAG'])
```

As this has not been a Randomized Control Experiment there has not really been a flag for control and treatment group. So let's introduce an artificial treatment effect. Imagine a new booking funnel, a landing page or an advertisement that the customer has been treated with. In our case we convert the variable Seat Category to this treatment indicator.


```python
try:
    df['treatment'] = df.SEAT_CATEGORY
    df = df.drop(columns = ['SEAT_CATEGORY'], axis=1)
except:
    print()
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8000 entries, 0 to 7999
    Data columns (total 15 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   index                8000 non-null   int64  
     1   Id                   8000 non-null   int64  
     2   PAXCOUNT             8000 non-null   int64  
     3   SALESCHANNEL         8000 non-null   object 
     4   TRIPTYPEDESC         8000 non-null   object 
     5   PURCHASELEAD         8000 non-null   int64  
     6   LENGTHOFSTAY         8000 non-null   int64  
     7   flight_hour          8000 non-null   int64  
     8   flight_day           8000 non-null   object 
     9   ROUTE                8000 non-null   object 
     10  geoNetwork_country   8000 non-null   object 
     11  BAGGAGE_CATEGORY     8000 non-null   int64  
     12  FNB_CATEGORY         8000 non-null   int64  
     13  flightDuration_hour  8000 non-null   float64
     14  treatment            8000 non-null   int64  
    dtypes: float64(1), int64(9), object(5)
    memory usage: 937.6+ KB
    

There is high correlation between treatment (i.e. hasCabin) and Class.
This is desirable in this case as it plays the role of the systematic factor affecting the treatment.
In a different context this could be a landing page on site that only specific visitors see.


```python
var = 'BAGGAGE_CATEGORY'
pd.pivot_table(df[['treatment',var,'Id']], \
               values = 'Id', index = 'treatment', columns = var, aggfunc= np.count_nonzero)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>BAGGAGE_CATEGORY</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2221</td>
      <td>3408</td>
    </tr>
    <tr>
      <th>1</th>
      <td>454</td>
      <td>1917</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>PAXCOUNT</th>
      <th>PURCHASELEAD</th>
      <th>LENGTHOFSTAY</th>
      <th>flight_hour</th>
      <th>BAGGAGE_CATEGORY</th>
      <th>FNB_CATEGORY</th>
      <th>flightDuration_hour</th>
      <th>treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>index</th>
      <td>1.000000</td>
      <td>0.005762</td>
      <td>0.010263</td>
      <td>0.012661</td>
      <td>0.018872</td>
      <td>0.009092</td>
      <td>0.000532</td>
      <td>0.003415</td>
      <td>0.006561</td>
      <td>0.004849</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0.005762</td>
      <td>1.000000</td>
      <td>0.135480</td>
      <td>0.015896</td>
      <td>-0.449091</td>
      <td>0.031731</td>
      <td>-0.174090</td>
      <td>-0.088515</td>
      <td>-0.259993</td>
      <td>-0.000516</td>
    </tr>
    <tr>
      <th>PAXCOUNT</th>
      <td>0.010263</td>
      <td>0.135480</td>
      <td>1.000000</td>
      <td>0.208453</td>
      <td>-0.113351</td>
      <td>0.002100</td>
      <td>0.117988</td>
      <td>0.036897</td>
      <td>-0.068885</td>
      <td>0.035038</td>
    </tr>
    <tr>
      <th>PURCHASELEAD</th>
      <td>0.012661</td>
      <td>0.015896</td>
      <td>0.208453</td>
      <td>1.000000</td>
      <td>-0.069222</td>
      <td>0.037607</td>
      <td>-0.017771</td>
      <td>-0.025227</td>
      <td>0.072581</td>
      <td>-0.013401</td>
    </tr>
    <tr>
      <th>LENGTHOFSTAY</th>
      <td>0.018872</td>
      <td>-0.449091</td>
      <td>-0.113351</td>
      <td>-0.069222</td>
      <td>1.000000</td>
      <td>-0.023447</td>
      <td>0.182585</td>
      <td>0.086383</td>
      <td>0.141260</td>
      <td>0.018458</td>
    </tr>
    <tr>
      <th>flight_hour</th>
      <td>0.009092</td>
      <td>0.031731</td>
      <td>0.002100</td>
      <td>0.037607</td>
      <td>-0.023447</td>
      <td>1.000000</td>
      <td>-0.017714</td>
      <td>0.002793</td>
      <td>-0.008462</td>
      <td>0.008569</td>
    </tr>
    <tr>
      <th>BAGGAGE_CATEGORY</th>
      <td>0.000532</td>
      <td>-0.174090</td>
      <td>0.117988</td>
      <td>-0.017771</td>
      <td>0.182585</td>
      <td>-0.017714</td>
      <td>1.000000</td>
      <td>0.219758</td>
      <td>0.051549</td>
      <td>0.196578</td>
    </tr>
    <tr>
      <th>FNB_CATEGORY</th>
      <td>0.003415</td>
      <td>-0.088515</td>
      <td>0.036897</td>
      <td>-0.025227</td>
      <td>0.086383</td>
      <td>0.002793</td>
      <td>0.219758</td>
      <td>1.000000</td>
      <td>0.137707</td>
      <td>0.310746</td>
    </tr>
    <tr>
      <th>flightDuration_hour</th>
      <td>0.006561</td>
      <td>-0.259993</td>
      <td>-0.068885</td>
      <td>0.072581</td>
      <td>0.141260</td>
      <td>-0.008462</td>
      <td>0.051549</td>
      <td>0.137707</td>
      <td>1.000000</td>
      <td>0.097837</td>
    </tr>
    <tr>
      <th>treatment</th>
      <td>0.004849</td>
      <td>-0.000516</td>
      <td>0.035038</td>
      <td>-0.013401</td>
      <td>0.018458</td>
      <td>0.008569</td>
      <td>0.196578</td>
      <td>0.310746</td>
      <td>0.097837</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Lets get rid of a few variables that do not much lineary correlate with the treatment variable and ideelay keep only those
variables that have an affect on the treatment variable.


```python
try:
    df = df.drop(columns = ['PAXCOUNT','PURCHASELEAD','LENGTHOFSTAY','flight_hour'], axis=1)
    df = df.drop(columns = ['ROUTE'], axis=1) # reduce compute resources
except:
    print()
```


```python
df.info()
df.sample()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8000 entries, 0 to 7999
    Data columns (total 10 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   index                8000 non-null   int64  
     1   Id                   8000 non-null   int64  
     2   SALESCHANNEL         8000 non-null   object 
     3   TRIPTYPEDESC         8000 non-null   object 
     4   flight_day           8000 non-null   object 
     5   geoNetwork_country   8000 non-null   object 
     6   BAGGAGE_CATEGORY     8000 non-null   int64  
     7   FNB_CATEGORY         8000 non-null   int64  
     8   flightDuration_hour  8000 non-null   float64
     9   treatment            8000 non-null   int64  
    dtypes: float64(1), int64(5), object(4)
    memory usage: 625.1+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>SALESCHANNEL</th>
      <th>TRIPTYPEDESC</th>
      <th>flight_day</th>
      <th>geoNetwork_country</th>
      <th>BAGGAGE_CATEGORY</th>
      <th>FNB_CATEGORY</th>
      <th>flightDuration_hour</th>
      <th>treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>184</th>
      <td>184</td>
      <td>43818</td>
      <td>Mobile</td>
      <td>RoundTrip</td>
      <td>Sun</td>
      <td>Thailand</td>
      <td>1</td>
      <td>1</td>
      <td>8.67</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Split the treatment variable from the rest of the covariates.


```python
T = df.treatment
X = df.loc[:,df.columns !='treatment']
X.shape
df.shape
T.shape
```




    (8000,)




```python
X_encoded = pd.get_dummies(X, columns = ['SALESCHANNEL', 'TRIPTYPEDESC','flight_day','geoNetwork_country'], \
                           prefix = {'SALESCHANNEL':'sc', 'TRIPTYPEDESC':'tt','flight_day':'fd','geoNetwork_country':'geo'}, drop_first=False)
```


```python
from sklearn.linear_model import LogisticRegression as lr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

# Design pipeline to build the treatment estimator
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_classifier', lr())
])

pipe.fit(X_encoded, T)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('logistic_classifier', LogisticRegression())])




```python
predictions = pipe.predict_proba(X_encoded)
predictions_binary = pipe.predict(X_encoded)
```


```python
print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))
```

    Accuracy: 0.7315
    
    Confusion matrix:
    [[4978  651]
     [1497  874]]
    
    F1 score is: 0.4487
    


```python
def logit(p):
    logit_value = math.log(p / (1-p))
    return logit_value
```

Convert propability to logit (based on the suggestion at https://youtu.be/gaUgW7NWai8?t=981)


```python
predictions_logit = np.array([logit(xi) for xi in predictions[:,1]])
```


```python
# Density distribution of propensity score (logic) broken down by treatment status

sns.set(rc={"figure.figsize":(14, 6)}) #width=8, height=4
fig, ax = plt.subplots(1,2)
fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
sns.kdeplot(x = predictions[:,1], hue = T , ax = ax[0])
ax[0].set_title('Propensity Score')
sns.kdeplot(x = predictions_logit, hue = T , ax = ax[1])
ax[1].axvline(-0, ls='--')
ax[1].set_title('Logit of Propensity Score')
#ax[1].set_xlim(-5,5)
plt.show()
```


    
![png](/blog/airasia-insurance_files/airasia-insurance_22_0.png)
    



```python
print(max(predictions_logit),min(predictions_logit))
```

    7.120447494606843 -9.159643967495708
    

The graph on the right shows the logit propensity scores = the probability distributions of both groups. We can see considerable overlap accross the range of values (-3,2). However on the left of "-3" the probabilities are a lot higher for non the non treatment group as expected. On the right side of zero, this is only marginally true.

For the propensity score matching this overlap is important in order to have balanced matching groups created, this nice overlap will make that work nicely.


```python
print(max(predictions_logit),min(predictions_logit))
common_support = (predictions_logit > -15) & (predictions_logit < 10)
```

    7.120447494606843 -9.159643967495708
    


```python
df.loc[:,'propensity_score'] = predictions[:,1]
df.loc[:,'propensity_score_logit'] = predictions_logit
df.loc[:,'outcome'] = y.INS_FLAG

X_encoded.loc[:,'propensity_score'] = predictions[:,1]
X_encoded.loc[:,'propensity_score_logit'] = predictions_logit
X_encoded.loc[:,'outcome'] = y.INS_FLAG
X_encoded.loc[:,'treatment'] = df.treatment
```

## Matching Implementation
Use Nearerst Neighbors to identify matching candidates. Then perform 1-to-1 matching by isolating/identifying groups of (T=1,T=0).
- Caliper: 25% of standart deviation of logit(propensity score)


```python
caliper = np.std(df.propensity_score) * 0.25

print('\nCaliper (radius) is: {:.4f}\n'.format(caliper))

df_data = X_encoded

knn = NearestNeighbors(n_neighbors=10 , p = 2, radius=caliper)
knn.fit(df_data[['propensity_score_logit']].to_numpy())
```

    
    Caliper (radius) is: 0.0427
    
    




    NearestNeighbors(n_neighbors=10, radius=0.04266206397392259)



For each data point (based on the logit propensity score) obtain (at most) 10 nearest matches. This is regardless of their treatment status.


```python
# Common support distances and indexes
distances , indexes = knn.kneighbors(
    df_data[['propensity_score_logit']].to_numpy(), \
    n_neighbors=10)
```


```python
print('For item 0, the 4 closest neighbors are (ignore first):')
for ds, idx  in zip(distances[0,0:4], indexes[0,0:4]):
    print('item index: {}'.format(idx))
    print('item distance: {:4f}'.format(ds))
print('...')
```

    For item 0, the 4 closest neighbors are (ignore first):
    item index: 0
    item distance: 0.000000
    item index: 5801
    item distance: 0.000156
    item index: 4512
    item distance: 0.000193
    item index: 7128
    item distance: 0.000220
    ...
    


```python
def propensity_score_matching(row, indexes, df_data):
    current_index = int(row['index']) # index column
    prop_score_logit = row['propensity_score_logit']
    for idx in indexes[current_index,:]:
        if (current_index != idx) and (row.treatment == 1) and (df_data.loc[idx].treatment == 0):
            return int(idx)
         
df_data['matched_element'] = df_data.reset_index().apply(propensity_score_matching, axis = 1, args = (indexes, df_data))
```


```python
treated_with_match = ~df_data.matched_element.isna()
```


```python

```


```python
treated_matched_data = df_data[treated_with_match][df_data.columns]
treated_matched_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>BAGGAGE_CATEGORY</th>
      <th>FNB_CATEGORY</th>
      <th>flightDuration_hour</th>
      <th>sc_Internet</th>
      <th>sc_Mobile</th>
      <th>tt_CircleTrip</th>
      <th>tt_OneWay</th>
      <th>tt_RoundTrip</th>
      <th>...</th>
      <th>geo_Turkey</th>
      <th>geo_United Arab Emirates</th>
      <th>geo_United Kingdom</th>
      <th>geo_United States</th>
      <th>geo_Vietnam</th>
      <th>propensity_score</th>
      <th>propensity_score_logit</th>
      <th>outcome</th>
      <th>treatment</th>
      <th>matched_element</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>10805</td>
      <td>0</td>
      <td>1</td>
      <td>8.83</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.295069</td>
      <td>-0.870892</td>
      <td>0</td>
      <td>1</td>
      <td>3045.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>24136</td>
      <td>1</td>
      <td>1</td>
      <td>8.58</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.562719</td>
      <td>0.252203</td>
      <td>0</td>
      <td>1</td>
      <td>7051.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>4798</td>
      <td>1</td>
      <td>0</td>
      <td>8.83</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.224072</td>
      <td>-1.242090</td>
      <td>0</td>
      <td>1</td>
      <td>5151.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 83 columns</p>
</div>




```python
cols = treated_matched_data.columns[0:-1]

def prop_matches(row, data, cols):
    return data.loc[row.matched_element][cols]
```


```python
%%time

untreated_matched_data = pd.DataFrame(data = treated_matched_data.matched_element)

"""attributes = ['Age', 'SibSp', 'Parch', 'Fare', 'sex_female', 'sex_male', 'embarked_C',
       'embarked_Q', 'embarked_S', 'class_1', 'class_2', 'class_3',
       'propensity_score', 'propensity_score_logit', 'outcome', 'treatment']
"""
for attr in cols:
    untreated_matched_data[attr] = untreated_matched_data.apply(prop_matches, axis = 1, data = df_data, cols = attr)
    
untreated_matched_data = untreated_matched_data.set_index('matched_element')
untreated_matched_data.head(3)
```

    Wall time: 26.4 s
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>BAGGAGE_CATEGORY</th>
      <th>FNB_CATEGORY</th>
      <th>flightDuration_hour</th>
      <th>sc_Internet</th>
      <th>sc_Mobile</th>
      <th>tt_CircleTrip</th>
      <th>tt_OneWay</th>
      <th>tt_RoundTrip</th>
      <th>...</th>
      <th>geo_Thailand</th>
      <th>geo_Turkey</th>
      <th>geo_United Arab Emirates</th>
      <th>geo_United Kingdom</th>
      <th>geo_United States</th>
      <th>geo_Vietnam</th>
      <th>propensity_score</th>
      <th>propensity_score_logit</th>
      <th>outcome</th>
      <th>treatment</th>
    </tr>
    <tr>
      <th>matched_element</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3045.0</th>
      <td>3045.0</td>
      <td>1417.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>8.83</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.295053</td>
      <td>-0.870966</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7051.0</th>
      <td>7051.0</td>
      <td>21235.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>8.83</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.562535</td>
      <td>0.251458</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5151.0</th>
      <td>5151.0</td>
      <td>23607.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.62</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.224075</td>
      <td>-1.242075</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 82 columns</p>
</div>




```python
untreated_matched_data.shape
```




    (2360, 82)




```python
treated_matched_data.shape
```




    (2360, 83)




```python
all_mached_data = pd.concat([treated_matched_data, untreated_matched_data]).reset_index()
```


```python
all_mached_data.index.is_unique
```




    True




```python
all_mached_data.treatment.value_counts()
```




    1.0    2360
    0.0    2360
    Name: treatment, dtype: int64



# Matching Review


```python
sns.set(rc={"figure.figsize":(14, 8)})
fig, ax = plt.subplots(2,1)
fig.suptitle('Comparison of {} split by outcome and treatment status'.format('propensity_score_logit'))
sns.stripplot(data = df_data.sample(n=1000), y = 'outcome', x = 'propensity_score_logit', 
              #alpha=0.7,
              hue = 'treatment', orient = 'h', ax = ax[0]).set(title = 'Before matching', xlim=(-4, 2))
sns.stripplot(data = all_mached_data.sample(n=1000), y = 'outcome', x = 'propensity_score_logit', 
              #alpha=0.7,
              hue = 'treatment', ax = ax[1] , orient = 'h').set(title = 'After matching', xlim=(-4, 2))
plt.subplots_adjust(hspace = 0.6)
plt.show()

```


    
![png](/blog/airasia-insurance_files/airasia-insurance_44_0.png)
    



```python
args = ['FNB_CATEGORY','BAGGAGE_CATEGORY','propensity_score_logit']

def plot(arg):
    fig, ax = plt.subplots(1,2)
    fig.suptitle('Comparison of {} split by treatment status.'.format(arg))
    sns.kdeplot(data = df_data, x = arg, hue = 'treatment', ax = ax[0]).set(title='Density before matching')
    sns.kdeplot(data = all_mached_data, x = arg, hue = 'treatment', ax = ax[1]).set(title='Density after matching')
    plt.show()

for arg in args:
    plot(arg)
```


    
![png](/blog/airasia-insurance_files/airasia-insurance_45_0.png)
    



    
![png](/blog/airasia-insurance_files/airasia-insurance_45_1.png)
    



    
![png](/blog/airasia-insurance_files/airasia-insurance_45_2.png)
    



```python
def cohenD (tmp, metricName):
    treated_metric = tmp[tmp.treatment == 1][metricName]
    untreated_metric = tmp[tmp.treatment == 0][metricName]
    
    d = ( treated_metric.mean() - untreated_metric.mean() ) / math.sqrt(((treated_metric.count()-1)*treated_metric.std()**2 + (untreated_metric.count()-1)*untreated_metric.std()**2) / (treated_metric.count() + untreated_metric.count()-2))
    return d

```


```python
%%time

data = []
cols = ['BAGGAGE_CATEGORY', 'FNB_CATEGORY',
       'flightDuration_hour', 'sc_Internet', 'sc_Mobile', 'tt_CircleTrip',
       'tt_OneWay', 'tt_RoundTrip', 'fd_Fri', 'fd_Mon', 'fd_Sat', 'fd_Sun',
       'fd_Thu', 'fd_Tue', 'fd_Wed', 'geo_Vietnam']
for cl in cols:
    data.append([cl,'before', cohenD(df_data,cl)])
    data.append([cl,'after', cohenD(all_mached_data,cl)])
```

    Wall time: 78 ms
    


```python
res = pd.DataFrame(data, columns=['variable','matching','effect_size'])
```


```python
sn_plot = sns.barplot(data = res, y = 'variable', x = 'effect_size', hue = 'matching', orient='h')
sn_plot.set(title='Standardised Mean differences accross covariates before and after matching')

```


    
![png](/blog/airasia-insurance_files/airasia-insurance_49_0.png)
    



```python
cols.append('treatment')
```

# Average Treatement effect


```python
overview = all_mached_data[['outcome','treatment']].groupby(by = ['treatment']).aggregate([np.mean, np.var, np.std, 'count'])
print(overview)
```

                outcome                          
                   mean       var       std count
    treatment                                    
    0.0        0.138559  0.119411  0.345559  2360
    1.0        0.178814  0.146902  0.383277  2360
    


```python
treated_outcome = overview['outcome']['mean'][1]
treated_counterfactual_outcome = overview['outcome']['mean'][0]
```


```python
att = treated_outcome - treated_counterfactual_outcome
print('The Average Treatment Effect (ATT): {:.4f}'.format(att))
```

    The Average Treatment Effect (ATT): 0.0403
    


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from causalinference import CausalModel

# y observed outcome, 
# T treatment indicator, 
# X covariate matrix (not one hot encoded),
causal = CausalModel(y.values, T, X)
print(causal.summary_stats)
```

    
    Summary Statistics
    
                          Controls (N_c=5629)        Treated (N_t=2371)             
           Variable         Mean         S.d.         Mean         S.d.     Raw-diff
    --------------------------------------------------------------------------------
                  Y        0.134        0.341        0.180        0.384        0.045
    
                          Controls (N_c=5629)        Treated (N_t=2371)             
           Variable         Mean         S.d.         Mean         S.d.     Nor-diff
    --------------------------------------------------------------------------------
                 X0     3992.233     2304.985     4016.753     2320.732        0.011
                 X1    25048.731    14472.131    25032.476    14163.444       -0.001
                 X2        0.605        0.489        0.809        0.394        0.458
                 X3        0.322        0.467        0.658        0.475        0.713
                 X4        7.171        1.524        7.493        1.421        0.218
    
    

From here we could continue with statistical testing to check if the lift/ treatment effect is significant statistically or we could answer that this minor effect we have noticed is practically not worth the effort and try a better advertisement or landing page.

## Conclusion


```python

```

---
title: Process Analytics & Process Mining with pm4py ~ Case study of an 'Order to Cash' process
tags: [Process Analytics, Process Mining, Business Processes, Petri Nets, BPMN, pm4py, Python, bupaR]
style: fill
color: info
description: I have a look at the pm4py python package of Fraunhofer Institute, which analog to R's bupaR package is able to mine and analyse processes through its log data. Analysing event data is an iterative process of three steps extraction, processing and analysis ...

---

I have a look at the [pm4py python package of Fraunhofer Institute](https://pm4py.fit.fraunhofer.de/), which analog to R's bupaR package is able to mine and analyse processes through its log data.

Analysing event data is an iterative process of three steps: extraction, processing and analysis.

There are three perspectives on a process and its analytics:

1. organizational perspective (focus on the actors)
2. control-flow perspective (focus on the flow and structuredness of the process)
3. performance perspective (focusses on time and efficiency)



Reminder: Each unique sequence of activities is called a trace or process variant.


```python
import os
os.environ["PATH"] += os.pathsep + 'C://Program Files//Graphviz//bin'
```


```python
import pandas as pd
import numpy as np
from datetime import date
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
```


```python
import pm4py
import pyreadr
```


```python
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
```

    pandas==1.3.4
    numpy==1.21.5
    seaborn==0.11.2
    pm4py==2.2.19.2
    pyreadr==0.4.4
    


```python
import sys
print('current env:' , sys.exec_prefix.split(os.sep)[-1] )
```

    current env: pm4py
    

We have got log data files for an order-to-cash process that we analyse end-to-end. 

At first we transform data from various sources to an common event log. 

After that we get a full overview picture of the process, explore the dimensions of the data and its different activities, stages and flows in the process.

Finally, we apply analysis tools to formulate an answer to a few questions of interest.

### Quotations Data (1/3)

It starts with preprocessing our process data from three log data sources. With some ETL and data wrangling we combine them finally in one large datafram.


```python
quotations = pyreadr.read_r('otc_quotations.RDS')
```


```python
quotations = quotations[None] 
```


```python
quotations.head()
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
      <th>quotation_id</th>
      <th>cancelled_at</th>
      <th>cancelled_by</th>
      <th>manufactContacted_at</th>
      <th>manufactContacted_by</th>
      <th>received_at</th>
      <th>received_by</th>
      <th>reminded_at</th>
      <th>reminded_by</th>
      <th>send_at</th>
      <th>send_by</th>
      <th>supplierContacted_at</th>
      <th>supplierContacted_by</th>
      <th>supplierOfferReceived_at</th>
      <th>supplierOfferReceived_by</th>
      <th>warehouseContacted_at</th>
      <th>warehouseContacted_by</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>2017-05-22 13:28:04</td>
      <td>Katherine</td>
      <td>2017-04-22 17:58:11</td>
      <td>Kimberly</td>
      <td>2017-04-16 20:34:12</td>
      <td>Andrea</td>
      <td>2017-05-14 19:06:41</td>
      <td>Jonathan</td>
      <td>2017-05-08 14:20:30</td>
      <td>Katherine</td>
      <td>2017-04-29 13:43:18</td>
      <td>Matthew</td>
      <td>2017-05-03 19:09:21</td>
      <td>Matthew</td>
      <td>2017-04-24 19:36:10</td>
      <td>Kimberly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-06-18 13:47:50</td>
      <td>Kimberly</td>
      <td>2017-06-09 11:19:31</td>
      <td>Jonathan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-02 18:50:58</td>
      <td>Andrea</td>
      <td>2017-06-20 12:19:31</td>
      <td>Kimberly</td>
      <td>2017-06-23 19:33:10</td>
      <td>Kimberly</td>
      <td>2017-06-15 19:30:07</td>
      <td>Kimberly</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-10-28 13:55:51</td>
      <td>Kimberly</td>
      <td>2017-10-14 18:55:47</td>
      <td>Jonathan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-11-09 11:27:11</td>
      <td>Andrea</td>
      <td>2017-10-26 18:06:29</td>
      <td>Kimberly</td>
      <td>2017-10-30 10:36:44</td>
      <td>Matthew</td>
      <td>2017-10-22 17:57:26</td>
      <td>Kimberly</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-09-08 13:29:05</td>
      <td>Jonathan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>2017-10-08 14:35:20</td>
      <td>Jonathan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-09-03 15:52:35</td>
      <td>Jonathan</td>
      <td>2017-10-02 16:25:02</td>
      <td>Jonathan</td>
      <td>2017-09-25 16:56:13</td>
      <td>Andrea</td>
      <td>2017-09-10 16:07:59</td>
      <td>Kimberly</td>
      <td>2017-09-14 14:37:03</td>
      <td>Matthew</td>
      <td>2017-09-16 17:14:54</td>
      <td>Matthew</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Inspect quotations
quotations.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1833 entries, 0 to 1832
    Data columns (total 17 columns):
     #   Column                    Non-Null Count  Dtype   
    ---  ------                    --------------  -----   
     0   quotation_id              1833 non-null   object  
     1   cancelled_at              255 non-null    object  
     2   cancelled_by              255 non-null    category
     3   manufactContacted_at      743 non-null    object  
     4   manufactContacted_by      743 non-null    category
     5   received_at               1833 non-null   object  
     6   received_by               1833 non-null   category
     7   reminded_at               504 non-null    object  
     8   reminded_by               504 non-null    category
     9   send_at                   1461 non-null   object  
     10  send_by                   1461 non-null   category
     11  supplierContacted_at      1099 non-null   object  
     12  supplierContacted_by      1099 non-null   category
     13  supplierOfferReceived_at  1099 non-null   object  
     14  supplierOfferReceived_by  1099 non-null   category
     15  warehouseContacted_at     741 non-null    object  
     16  warehouseContacted_by     741 non-null    category
    dtypes: category(8), object(9)
    memory usage: 144.3+ KB
    


```python
offer_history = quotations.melt(id_vars='quotation_id')
offer_history
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
      <th>quotation_id</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>cancelled_at</td>
      <td>2017-05-22 13:28:04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>cancelled_at</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>cancelled_at</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>cancelled_at</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>cancelled_at</td>
      <td>2017-10-08 14:35:20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29323</th>
      <td>quo-985</td>
      <td>warehouseContacted_by</td>
      <td>Kimberly</td>
    </tr>
    <tr>
      <th>29324</th>
      <td>quo-986</td>
      <td>warehouseContacted_by</td>
      <td>Matthew</td>
    </tr>
    <tr>
      <th>29325</th>
      <td>quo-987</td>
      <td>warehouseContacted_by</td>
      <td>Matthew</td>
    </tr>
    <tr>
      <th>29326</th>
      <td>quo-994</td>
      <td>warehouseContacted_by</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29327</th>
      <td>quo-996</td>
      <td>warehouseContacted_by</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>29328 rows × 3 columns</p>
</div>




```python
offer_history[['activity','info']] = offer_history.variable.str.split('_', expand=True)
offer_history.drop('variable', axis=1, inplace=True)
offer_history = offer_history[['quotation_id', 'activity', 'info', 'value']]
```


```python
offer_history
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
      <th>quotation_id</th>
      <th>activity</th>
      <th>info</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>cancelled</td>
      <td>at</td>
      <td>2017-05-22 13:28:04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>cancelled</td>
      <td>at</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>cancelled</td>
      <td>at</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>cancelled</td>
      <td>at</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>cancelled</td>
      <td>at</td>
      <td>2017-10-08 14:35:20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29323</th>
      <td>quo-985</td>
      <td>warehouseContacted</td>
      <td>by</td>
      <td>Kimberly</td>
    </tr>
    <tr>
      <th>29324</th>
      <td>quo-986</td>
      <td>warehouseContacted</td>
      <td>by</td>
      <td>Matthew</td>
    </tr>
    <tr>
      <th>29325</th>
      <td>quo-987</td>
      <td>warehouseContacted</td>
      <td>by</td>
      <td>Matthew</td>
    </tr>
    <tr>
      <th>29326</th>
      <td>quo-994</td>
      <td>warehouseContacted</td>
      <td>by</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29327</th>
      <td>quo-996</td>
      <td>warehouseContacted</td>
      <td>by</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>29328 rows × 4 columns</p>
</div>




```python
recode = {"info":  {"at": "timestamp", "by": "resource"}}
offer_history_recode = offer_history.replace(recode)
offer_history_recode['info'] = offer_history_recode['info'].astype('category')
offer_history_recode
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
      <th>quotation_id</th>
      <th>activity</th>
      <th>info</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>cancelled</td>
      <td>timestamp</td>
      <td>2017-05-22 13:28:04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>cancelled</td>
      <td>timestamp</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>cancelled</td>
      <td>timestamp</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>cancelled</td>
      <td>timestamp</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>cancelled</td>
      <td>timestamp</td>
      <td>2017-10-08 14:35:20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29323</th>
      <td>quo-985</td>
      <td>warehouseContacted</td>
      <td>resource</td>
      <td>Kimberly</td>
    </tr>
    <tr>
      <th>29324</th>
      <td>quo-986</td>
      <td>warehouseContacted</td>
      <td>resource</td>
      <td>Matthew</td>
    </tr>
    <tr>
      <th>29325</th>
      <td>quo-987</td>
      <td>warehouseContacted</td>
      <td>resource</td>
      <td>Matthew</td>
    </tr>
    <tr>
      <th>29326</th>
      <td>quo-994</td>
      <td>warehouseContacted</td>
      <td>resource</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29327</th>
      <td>quo-996</td>
      <td>warehouseContacted</td>
      <td>resource</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>29328 rows × 4 columns</p>
</div>




```python
offer_history_spread = offer_history_recode.pivot( index=['quotation_id', 'activity'], columns='info', values='value').reset_index()
```


```python
offer_history_spread
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
      <th>info</th>
      <th>quotation_id</th>
      <th>activity</th>
      <th>resource</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>cancelled</td>
      <td>Katherine</td>
      <td>2017-05-22 13:28:04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1003</td>
      <td>manufactContacted</td>
      <td>Kimberly</td>
      <td>2017-04-22 17:58:11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1003</td>
      <td>received</td>
      <td>Andrea</td>
      <td>2017-04-16 20:34:12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1003</td>
      <td>reminded</td>
      <td>Jonathan</td>
      <td>2017-05-14 19:06:41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1003</td>
      <td>send</td>
      <td>Katherine</td>
      <td>2017-05-08 14:20:30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14659</th>
      <td>quo-996</td>
      <td>reminded</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14660</th>
      <td>quo-996</td>
      <td>send</td>
      <td>Andrea</td>
      <td>2017-01-30 18:53:32</td>
    </tr>
    <tr>
      <th>14661</th>
      <td>quo-996</td>
      <td>supplierContacted</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14662</th>
      <td>quo-996</td>
      <td>supplierOfferReceived</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14663</th>
      <td>quo-996</td>
      <td>warehouseContacted</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>14664 rows × 4 columns</p>
</div>



### Validations Data (2/3)


```python
validations = pyreadr.read_r('otc_validations.RDS')
```


```python
validations = validations[None]
validations
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
      <th>quotation_id</th>
      <th>resource</th>
      <th>started</th>
      <th>completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>Jonathan</td>
      <td>2017-04-17 14:59:08</td>
      <td>2017-04-19 18:32:57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>Andrea</td>
      <td>2017-06-11 13:10:45</td>
      <td>2017-06-13 12:18:57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>Katherine</td>
      <td>2017-10-16 15:59:18</td>
      <td>2017-10-18 16:21:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>Andrea</td>
      <td>2017-09-09 17:58:39</td>
      <td>2017-09-12 20:58:14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>Andrea</td>
      <td>2017-09-05 17:40:32</td>
      <td>2017-09-08 15:48:41</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1828</th>
      <td>quo-985</td>
      <td>Andrea</td>
      <td>2017-03-29 19:15:36</td>
      <td>2017-04-01 14:33:23</td>
    </tr>
    <tr>
      <th>1829</th>
      <td>quo-986</td>
      <td>Katherine</td>
      <td>2017-10-23 11:28:02</td>
      <td>2017-10-26 14:49:49</td>
    </tr>
    <tr>
      <th>1830</th>
      <td>quo-987</td>
      <td>Katherine</td>
      <td>2017-02-14 16:27:52</td>
      <td>2017-02-17 11:33:17</td>
    </tr>
    <tr>
      <th>1831</th>
      <td>quo-994</td>
      <td>Andrea</td>
      <td>2017-09-12 18:37:27</td>
      <td>2017-09-13 15:25:31</td>
    </tr>
    <tr>
      <th>1832</th>
      <td>quo-996</td>
      <td>Katherine</td>
      <td>2017-01-17 18:16:26</td>
      <td>2017-01-18 18:42:02</td>
    </tr>
  </tbody>
</table>
<p>1833 rows × 4 columns</p>
</div>




```python
validate_history = validations
validate_history['activity'] = "Validate"
validate_history['action'] = validate_history.quotation_id + '-validate'
validate_history
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
      <th>quotation_id</th>
      <th>resource</th>
      <th>started</th>
      <th>completed</th>
      <th>activity</th>
      <th>action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>Jonathan</td>
      <td>2017-04-17 14:59:08</td>
      <td>2017-04-19 18:32:57</td>
      <td>Validate</td>
      <td>quo-1003-validate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>Andrea</td>
      <td>2017-06-11 13:10:45</td>
      <td>2017-06-13 12:18:57</td>
      <td>Validate</td>
      <td>quo-1004-validate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>Katherine</td>
      <td>2017-10-16 15:59:18</td>
      <td>2017-10-18 16:21:56</td>
      <td>Validate</td>
      <td>quo-1006-validate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>Andrea</td>
      <td>2017-09-09 17:58:39</td>
      <td>2017-09-12 20:58:14</td>
      <td>Validate</td>
      <td>quo-1008-validate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>Andrea</td>
      <td>2017-09-05 17:40:32</td>
      <td>2017-09-08 15:48:41</td>
      <td>Validate</td>
      <td>quo-1010-validate</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1828</th>
      <td>quo-985</td>
      <td>Andrea</td>
      <td>2017-03-29 19:15:36</td>
      <td>2017-04-01 14:33:23</td>
      <td>Validate</td>
      <td>quo-985-validate</td>
    </tr>
    <tr>
      <th>1829</th>
      <td>quo-986</td>
      <td>Katherine</td>
      <td>2017-10-23 11:28:02</td>
      <td>2017-10-26 14:49:49</td>
      <td>Validate</td>
      <td>quo-986-validate</td>
    </tr>
    <tr>
      <th>1830</th>
      <td>quo-987</td>
      <td>Katherine</td>
      <td>2017-02-14 16:27:52</td>
      <td>2017-02-17 11:33:17</td>
      <td>Validate</td>
      <td>quo-987-validate</td>
    </tr>
    <tr>
      <th>1831</th>
      <td>quo-994</td>
      <td>Andrea</td>
      <td>2017-09-12 18:37:27</td>
      <td>2017-09-13 15:25:31</td>
      <td>Validate</td>
      <td>quo-994-validate</td>
    </tr>
    <tr>
      <th>1832</th>
      <td>quo-996</td>
      <td>Katherine</td>
      <td>2017-01-17 18:16:26</td>
      <td>2017-01-18 18:42:02</td>
      <td>Validate</td>
      <td>quo-996-validate</td>
    </tr>
  </tbody>
</table>
<p>1833 rows × 6 columns</p>
</div>




```python
validate_history_gathered = validate_history.melt(id_vars=['quotation_id','resource', 'activity','action'],
                                                  var_name='lifecycle', value_name='timestamp'
                                                 )
```


```python
validate_history_gathered
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
      <th>quotation_id</th>
      <th>resource</th>
      <th>activity</th>
      <th>action</th>
      <th>lifecycle</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>Jonathan</td>
      <td>Validate</td>
      <td>quo-1003-validate</td>
      <td>started</td>
      <td>2017-04-17 14:59:08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1004-validate</td>
      <td>started</td>
      <td>2017-06-11 13:10:45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-1006-validate</td>
      <td>started</td>
      <td>2017-10-16 15:59:18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1008-validate</td>
      <td>started</td>
      <td>2017-09-09 17:58:39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1010-validate</td>
      <td>started</td>
      <td>2017-09-05 17:40:32</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3661</th>
      <td>quo-985</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-985-validate</td>
      <td>completed</td>
      <td>2017-04-01 14:33:23</td>
    </tr>
    <tr>
      <th>3662</th>
      <td>quo-986</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-986-validate</td>
      <td>completed</td>
      <td>2017-10-26 14:49:49</td>
    </tr>
    <tr>
      <th>3663</th>
      <td>quo-987</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-987-validate</td>
      <td>completed</td>
      <td>2017-02-17 11:33:17</td>
    </tr>
    <tr>
      <th>3664</th>
      <td>quo-994</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-994-validate</td>
      <td>completed</td>
      <td>2017-09-13 15:25:31</td>
    </tr>
    <tr>
      <th>3665</th>
      <td>quo-996</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-996-validate</td>
      <td>completed</td>
      <td>2017-01-18 18:42:02</td>
    </tr>
  </tbody>
</table>
<p>3666 rows × 6 columns</p>
</div>




```python
recode = {"lifecycle":  {"started": "start", "completed": "complete"}}
validate_history_recode = validate_history_gathered.replace(recode)
validate_history_recode['lifecycle'] = validate_history_recode['lifecycle'].astype('category')
validate_history_recode
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
      <th>quotation_id</th>
      <th>resource</th>
      <th>activity</th>
      <th>action</th>
      <th>lifecycle</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>Jonathan</td>
      <td>Validate</td>
      <td>quo-1003-validate</td>
      <td>start</td>
      <td>2017-04-17 14:59:08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1004-validate</td>
      <td>start</td>
      <td>2017-06-11 13:10:45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-1006-validate</td>
      <td>start</td>
      <td>2017-10-16 15:59:18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1008-validate</td>
      <td>start</td>
      <td>2017-09-09 17:58:39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1010-validate</td>
      <td>start</td>
      <td>2017-09-05 17:40:32</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3661</th>
      <td>quo-985</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-985-validate</td>
      <td>complete</td>
      <td>2017-04-01 14:33:23</td>
    </tr>
    <tr>
      <th>3662</th>
      <td>quo-986</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-986-validate</td>
      <td>complete</td>
      <td>2017-10-26 14:49:49</td>
    </tr>
    <tr>
      <th>3663</th>
      <td>quo-987</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-987-validate</td>
      <td>complete</td>
      <td>2017-02-17 11:33:17</td>
    </tr>
    <tr>
      <th>3664</th>
      <td>quo-994</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-994-validate</td>
      <td>complete</td>
      <td>2017-09-13 15:25:31</td>
    </tr>
    <tr>
      <th>3665</th>
      <td>quo-996</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-996-validate</td>
      <td>complete</td>
      <td>2017-01-18 18:42:02</td>
    </tr>
  </tbody>
</table>
<p>3666 rows × 6 columns</p>
</div>



bring both dataframes on the same variable names and with identical lifecycle encoding


```python
offer_history_recode = offer_history_spread
offer_history_recode['lifecycle'] = "complete"
offer_history_recode['increment_num'] = range(0+1, len(offer_history_recode.quotation_id)+1)
offer_history_recode['action'] = offer_history_recode.quotation_id.astype(str) + '-' + offer_history_recode.increment_num.astype(str)
offer_history_recode.drop('increment_num', axis=1, inplace=True)
```


```python
offer_history_recode = offer_history_recode.dropna() # to arrive at 7735 rows - undocumented in case study
```


```python
# concat quotations + validations histories to sales history
```


```python
sales_history2 = pd.concat([offer_history_recode,validate_history_recode])
sales_history2
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
      <th>quotation_id</th>
      <th>activity</th>
      <th>resource</th>
      <th>timestamp</th>
      <th>lifecycle</th>
      <th>action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>cancelled</td>
      <td>Katherine</td>
      <td>2017-05-22 13:28:04</td>
      <td>complete</td>
      <td>quo-1003-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1003</td>
      <td>manufactContacted</td>
      <td>Kimberly</td>
      <td>2017-04-22 17:58:11</td>
      <td>complete</td>
      <td>quo-1003-2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1003</td>
      <td>received</td>
      <td>Andrea</td>
      <td>2017-04-16 20:34:12</td>
      <td>complete</td>
      <td>quo-1003-3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1003</td>
      <td>reminded</td>
      <td>Jonathan</td>
      <td>2017-05-14 19:06:41</td>
      <td>complete</td>
      <td>quo-1003-4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1003</td>
      <td>send</td>
      <td>Katherine</td>
      <td>2017-05-08 14:20:30</td>
      <td>complete</td>
      <td>quo-1003-5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3661</th>
      <td>quo-985</td>
      <td>Validate</td>
      <td>Andrea</td>
      <td>2017-04-01 14:33:23</td>
      <td>complete</td>
      <td>quo-985-validate</td>
    </tr>
    <tr>
      <th>3662</th>
      <td>quo-986</td>
      <td>Validate</td>
      <td>Katherine</td>
      <td>2017-10-26 14:49:49</td>
      <td>complete</td>
      <td>quo-986-validate</td>
    </tr>
    <tr>
      <th>3663</th>
      <td>quo-987</td>
      <td>Validate</td>
      <td>Katherine</td>
      <td>2017-02-17 11:33:17</td>
      <td>complete</td>
      <td>quo-987-validate</td>
    </tr>
    <tr>
      <th>3664</th>
      <td>quo-994</td>
      <td>Validate</td>
      <td>Andrea</td>
      <td>2017-09-13 15:25:31</td>
      <td>complete</td>
      <td>quo-994-validate</td>
    </tr>
    <tr>
      <th>3665</th>
      <td>quo-996</td>
      <td>Validate</td>
      <td>Katherine</td>
      <td>2017-01-18 18:42:02</td>
      <td>complete</td>
      <td>quo-996-validate</td>
    </tr>
  </tbody>
</table>
<p>11401 rows × 6 columns</p>
</div>



### Sales Data (3/3)


```python
sales_history = pyreadr.read_r('otc_sales_history.RDS')
sales_history = sales_history[None]
sales_history
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
      <th>quotation_id</th>
      <th>resource</th>
      <th>activity</th>
      <th>action</th>
      <th>lifecycle</th>
      <th>timestamp</th>
      <th>sales_order_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quo-1003</td>
      <td>Jonathan</td>
      <td>Validate</td>
      <td>quo-1003-validate</td>
      <td>start</td>
      <td>2017-04-17 14:59:08</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>quo-1004</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1004-validate</td>
      <td>start</td>
      <td>2017-06-11 13:10:45</td>
      <td>order-17-56548</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quo-1006</td>
      <td>Katherine</td>
      <td>Validate</td>
      <td>quo-1006-validate</td>
      <td>start</td>
      <td>2017-10-16 15:59:18</td>
      <td>order-17-56550</td>
    </tr>
    <tr>
      <th>3</th>
      <td>quo-1008</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1008-validate</td>
      <td>start</td>
      <td>2017-09-09 17:58:39</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>quo-1010</td>
      <td>Andrea</td>
      <td>Validate</td>
      <td>quo-1010-validate</td>
      <td>start</td>
      <td>2017-09-05 17:40:32</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14690</th>
      <td>quo-929</td>
      <td>Katherine</td>
      <td>Decline Request For Quotation</td>
      <td>quo-929-decline</td>
      <td>complete</td>
      <td>2017-04-21 15:32:49</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14691</th>
      <td>quo-930</td>
      <td>Andrea</td>
      <td>Decline Request For Quotation</td>
      <td>quo-930-decline</td>
      <td>complete</td>
      <td>2017-10-29 18:07:52</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14692</th>
      <td>quo-969</td>
      <td>Katherine</td>
      <td>Decline Request For Quotation</td>
      <td>quo-969-decline</td>
      <td>complete</td>
      <td>2017-06-03 11:59:36</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14693</th>
      <td>quo-983</td>
      <td>Katherine</td>
      <td>Decline Request For Quotation</td>
      <td>quo-983-decline</td>
      <td>complete</td>
      <td>2017-01-18 15:42:26</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14694</th>
      <td>quo-994</td>
      <td>Andrea</td>
      <td>Decline Request For Quotation</td>
      <td>quo-994-decline</td>
      <td>complete</td>
      <td>2017-09-17 17:46:38</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>14695 rows × 7 columns</p>
</div>




```python
order_history = pyreadr.read_r('otc_order_history.RDS')
order_history = order_history[None]
order_history.rename(columns = {'time':'timestamp'}, inplace = True)
order_history
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
      <th>sales_order_id</th>
      <th>action</th>
      <th>activity</th>
      <th>resource</th>
      <th>status</th>
      <th>timestamp</th>
      <th>activity_cost</th>
      <th>quotation_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>order-17-56542</td>
      <td>order-17-56542-0000001</td>
      <td>Receive Sales Order</td>
      <td>Katherine</td>
      <td>start</td>
      <td>2017-10-17 12:37:22</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>order-17-56542</td>
      <td>order-17-56542-0000002</td>
      <td>Send To Manufacturing</td>
      <td>Jonathan</td>
      <td>start</td>
      <td>2017-10-19 15:30:40</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>order-17-56543</td>
      <td>order-17-56543-0000003</td>
      <td>Receive Sales Order</td>
      <td>Andrea</td>
      <td>start</td>
      <td>2017-11-04 15:27:23</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>order-17-56543</td>
      <td>order-17-56543-0000004</td>
      <td>Send To Manufacturing</td>
      <td>Jonathan</td>
      <td>start</td>
      <td>2017-11-13 15:08:58</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>order-17-56544</td>
      <td>order-17-56544-0000005</td>
      <td>Receive Sales Order</td>
      <td>Andrea</td>
      <td>start</td>
      <td>2017-04-08 15:10:53</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>60799</th>
      <td>order-17-59561</td>
      <td>order-17-59561-0060800</td>
      <td>Prepare Invoice</td>
      <td>Virginia</td>
      <td>complete</td>
      <td>2017-03-09 12:21:34</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60800</th>
      <td>order-17-59562</td>
      <td>order-17-59562-0060801</td>
      <td>Prepare Invoice</td>
      <td>Virginia</td>
      <td>complete</td>
      <td>2017-03-26 19:26:07</td>
      <td>NaN</td>
      <td>quo-996</td>
    </tr>
    <tr>
      <th>60801</th>
      <td>order-17-59563</td>
      <td>order-17-59563-0060802</td>
      <td>Prepare Invoice</td>
      <td>George</td>
      <td>complete</td>
      <td>2017-11-24 18:46:48</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60802</th>
      <td>order-17-59564</td>
      <td>order-17-59564-0060803</td>
      <td>Prepare Invoice</td>
      <td>George</td>
      <td>complete</td>
      <td>2017-03-04 10:30:07</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60803</th>
      <td>order-17-59565</td>
      <td>order-17-59565-0060804</td>
      <td>Prepare Invoice</td>
      <td>George</td>
      <td>complete</td>
      <td>2017-05-10 14:02:24</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>60804 rows × 8 columns</p>
</div>




```python
otc = pd.concat([sales_history, order_history])
#otc[otc.activity == 'Receive Sales Order']
otc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 75499 entries, 0 to 60803
    Data columns (total 9 columns):
     #   Column          Non-Null Count  Dtype   
    ---  ------          --------------  -----   
     0   quotation_id    38922 non-null  object  
     1   resource        75499 non-null  object  
     2   activity        75499 non-null  object  
     3   action          75499 non-null  object  
     4   lifecycle       14695 non-null  object  
     5   timestamp       75499 non-null  object  
     6   sales_order_id  71321 non-null  object  
     7   status          60804 non-null  category
     8   activity_cost   14100 non-null  float64 
    dtypes: category(1), float64(1), object(7)
    memory usage: 5.3+ MB
    


```python
otc_event_log = otc.copy()
otc_event_log["timestamp"] = pd.to_datetime(otc_event_log["timestamp"])
otc_event_log = otc_event_log.sort_values(by=["timestamp"]) # https://aginic.com/blog/your-guide-to-process-mining/
otc_event_log['case_id'] = otc_event_log.quotation_id.fillna('') + '-' + otc_event_log.sales_order_id.fillna('')
#df['new_col'] = df['foodstuff'].fillna('') + df['type'].fillna('')
#otc_event_log['case_id'] = otc_event_log['quotation_id'].combine(otc_event_log['sales_order_id'], lambda a, b: ((a or "") + (b or "")) or None, None)
otc_event_log
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
      <th>quotation_id</th>
      <th>resource</th>
      <th>activity</th>
      <th>action</th>
      <th>lifecycle</th>
      <th>timestamp</th>
      <th>sales_order_id</th>
      <th>status</th>
      <th>activity_cost</th>
      <th>case_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10949</th>
      <td>quo-808</td>
      <td>Katherine</td>
      <td>Receive Request For Quotation</td>
      <td>quo-808-13875</td>
      <td>complete</td>
      <td>2017-01-01 12:13:56</td>
      <td>order-17-59385</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>quo-808-order-17-59385</td>
    </tr>
    <tr>
      <th>9936</th>
      <td>quo-3636</td>
      <td>Andrea</td>
      <td>Receive Request For Quotation</td>
      <td>quo-3636-11915</td>
      <td>complete</td>
      <td>2017-01-01 12:28:35</td>
      <td>order-17-58955</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>quo-3636-order-17-58955</td>
    </tr>
    <tr>
      <th>11047</th>
      <td>quo-846</td>
      <td>Katherine</td>
      <td>Receive Request For Quotation</td>
      <td>quo-846-14035</td>
      <td>complete</td>
      <td>2017-01-01 12:37:46</td>
      <td>order-17-59422</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>quo-846-order-17-59422</td>
    </tr>
    <tr>
      <th>3020</th>
      <td>NaN</td>
      <td>Jonathan</td>
      <td>Receive Sales Order</td>
      <td>order-17-58052-0003021</td>
      <td>NaN</td>
      <td>2017-01-01 13:55:11</td>
      <td>order-17-58052</td>
      <td>start</td>
      <td>NaN</td>
      <td>-order-17-58052</td>
    </tr>
    <tr>
      <th>4370</th>
      <td>NaN</td>
      <td>Jonathan</td>
      <td>Receive Sales Order</td>
      <td>order-17-58727-0004371</td>
      <td>NaN</td>
      <td>2017-01-01 14:32:18</td>
      <td>order-17-58727</td>
      <td>start</td>
      <td>NaN</td>
      <td>-order-17-58727</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47988</th>
      <td>quo-1186</td>
      <td>George</td>
      <td>Send Last Reminder</td>
      <td>order-17-56717-0047989</td>
      <td>NaN</td>
      <td>2018-04-16 15:08:27</td>
      <td>order-17-56717</td>
      <td>start</td>
      <td>NaN</td>
      <td>quo-1186-order-17-56717</td>
    </tr>
    <tr>
      <th>50151</th>
      <td>quo-1879</td>
      <td>Virginia</td>
      <td>Payment Received</td>
      <td>order-17-57350-0050152</td>
      <td>NaN</td>
      <td>2018-04-17 13:09:40</td>
      <td>order-17-57350</td>
      <td>start</td>
      <td>NaN</td>
      <td>quo-1879-order-17-57350</td>
    </tr>
    <tr>
      <th>47989</th>
      <td>quo-1186</td>
      <td>George</td>
      <td>Payment Received</td>
      <td>order-17-56717-0047990</td>
      <td>NaN</td>
      <td>2018-04-19 14:00:10</td>
      <td>order-17-56717</td>
      <td>start</td>
      <td>NaN</td>
      <td>quo-1186-order-17-56717</td>
    </tr>
    <tr>
      <th>30137</th>
      <td>quo-2660</td>
      <td>Tina</td>
      <td>Fix Goods</td>
      <td>order-17-58065-0030138</td>
      <td>NaN</td>
      <td>2018-04-20 10:12:04</td>
      <td>order-17-58065</td>
      <td>complete</td>
      <td>NaN</td>
      <td>quo-2660-order-17-58065</td>
    </tr>
    <tr>
      <th>52625</th>
      <td>quo-2660</td>
      <td>George</td>
      <td>Payment Received</td>
      <td>order-17-58065-0052626</td>
      <td>NaN</td>
      <td>2018-04-22 19:56:13</td>
      <td>order-17-58065</td>
      <td>start</td>
      <td>NaN</td>
      <td>quo-2660-order-17-58065</td>
    </tr>
  </tbody>
</table>
<p>75499 rows × 10 columns</p>
</div>




```python
len(otc_event_log.sales_order_id.unique())
```




    3025




```python
# only slice cases with sales orders
otc_event_log_only_so = otc_event_log[~otc_event_log.sales_order_id.isnull()]
otc_event_log_only_so.sales_order_id.isnull().sum()
```




    0




```python
# pm4py needs for later applied methods caseid column in the first column of the dataframe

# get rid of unused cols
otc_event_log_only_so2 = otc_event_log_only_so.copy().drop(['lifecycle', 'status'], axis=1)

first_column_to_be = otc_event_log_only_so2.pop('sales_order_id')
otc_event_log_only_so2.insert(0, 'sales_order_id', first_column_to_be)
  
otc_event_log_only_so2.shape
```




    (71321, 8)




```python
#otc_event_log[otc_event_log.activity == 'Receive Request For Quotation']
```


```python
from pm4py.objects.conversion.log import converter as log_converter

#parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'sales_order_id'}
#parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.ACTIVITY_KEY: 'activity'}
#parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.TIMESTAMP_KEY: 'timestamp'}
#event_log = log_converter.apply(otc, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

event_log = pm4py.format_dataframe(otc_event_log_only_so2, case_id='sales_order_id', activity_key='activity', timestamp_key='timestamp')
```


```python
type(event_log)
```




    pandas.core.frame.DataFrame




```python
start_activities = pm4py.get_start_activities(event_log)
end_activities = pm4py.get_end_activities(event_log)
print("Start activities: {}\nEnd activities: {}".format(start_activities, end_activities))
```

    Start activities: {'Receive Sales Order': 1818, 'Receive Request For Quotation': 1206}
    End activities: {'Payment Received': 2564, 'Create Credit Note': 204, 'Send Invoice': 175, 'Submit To Credit Collection': 81}
    


```python
# process mining 
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

# viz
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization

# misc 
from pm4py.objects.conversion.process_tree import converter as pt_converter

# alpha miner
net, initial_marking, final_marking = alpha_miner.apply(event_log)
```


```python
# Visualise
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
```

    C:\Users\SD\anaconda3\envs\pm4py\lib\site-packages\ipykernel_launcher.py:2: DeprecatedWarning: apply is deprecated as of 2.2.5 and will be removed in 3.0.0. please use pm4py.visualization.petri_net.algorithm instead
      
    


    
![png](case-study-process-mining-process-analytics-order-to-cash-process-blog_files/case-study-process-mining-process-analytics-order-to-cash-process-blog_44_1.png)
    



```python
# heuristics miner
heu_net = heuristics_miner.apply_heu(event_log)

# viz
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)
```


    
![png](case-study-process-mining-process-analytics-order-to-cash-process-blog_files/case-study-process-mining-process-analytics-order-to-cash-process-blog_45_0.png)
    



```python
from pm4py.algo.filtering.log.start_activities import start_activities_filter
log_start = start_activities_filter.get_start_activities(event_log)
log_start
```




    {}




```python
type(event_log)
```




    pandas.core.frame.DataFrame




```python
event_log.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 71321 entries, 0 to 57779
    Data columns (total 13 columns):
     #   Column             Non-Null Count  Dtype         
    ---  ------             --------------  -----         
     0   sales_order_id     71321 non-null  object        
     1   quotation_id       34744 non-null  object        
     2   resource           71321 non-null  object        
     3   activity           71321 non-null  object        
     4   action             71321 non-null  object        
     5   timestamp          71321 non-null  datetime64[ns]
     6   activity_cost      14100 non-null  float64       
     7   case_id            71321 non-null  object        
     8   case:concept:name  71321 non-null  string        
     9   concept:name       71321 non-null  string        
     10  time:timestamp     71321 non-null  datetime64[ns]
     11  @@index            71321 non-null  int64         
     12  start_timestamp    71321 non-null  datetime64[ns]
    dtypes: datetime64[ns](3), float64(1), int64(1), object(6), string(2)
    memory usage: 7.6+ MB
    


```python
process_tree = pm4py.discover_process_tree_inductive(event_log)
pm4py.view_process_tree(process_tree, #bgcolor='white'
                        #parameters={'bgcolor': 'white'}
                       )
```


    
![png](case-study-process-mining-process-analytics-order-to-cash-process-blog_files/case-study-process-mining-process-analytics-order-to-cash-process-blog_49_0.png)
    



```python
otc_event_log_only_so = pm4py.format_dataframe(otc_event_log_only_so.copy(), case_id='sales_order_id', activity_key='activity', timestamp_key='timestamp')
```


```python
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
dfg = dfg_discovery.apply(event_log)

from pm4py.visualization.dfg import visualizer as dfg_visualization
gviz = dfg_visualization.apply(dfg, log=event_log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)
```


    
![png](case-study-process-mining-process-analytics-order-to-cash-process-blog_files/case-study-process-mining-process-analytics-order-to-cash-process-blog_51_0.png)
    



```python
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

dfg = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.PERFORMANCE)
gviz = dfg_visualization.apply(dfg, log=event_log, variant=dfg_visualization.Variants.PERFORMANCE)
dfg_visualization.view(gviz)
```


    
![png](case-study-process-mining-process-analytics-order-to-cash-process-blog_files/case-study-process-mining-process-analytics-order-to-cash-process-blog_52_0.png)
    


#### Filtering for subprocess 'Payment' ("Send Invoice", "Payment Received")

We are interested in the subprocess payment and its cycle time, that may be in focus for a recent performance project and lets see if and how pm4py can help us get insight.


```python
# drop some unneeeded cols
otc_event_log_only_so2 = otc_event_log_only_so.drop(['lifecycle', 'status'], axis=1)

# undocumented(?) pm4py requirement: column sequence in df is important
first_column_to_be = otc_event_log_only_so2.pop('sales_order_id')
otc_event_log_only_so2.insert(0, 'sales_order_id', first_column_to_be)
otc_event_log_only_so2.shape
```




    (71321, 12)




```python
#parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'} # identify the case_id_key name (if not change it will simply be the mane of the coloumn)
#event_log_unfiltered = log_converter.apply(otc_event_log_only_so2, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
```


```python
#otc_event_log_only_so2.head()
```

Looking at all activities of `order-17-56542`:


```python
# before 'between filter'
otc_event_log_only_so2[otc_event_log_only_so2['case:concept:name'] == 'order-17-56542'].sort_values(by=['sales_order_id', 'timestamp'])
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
      <th>sales_order_id</th>
      <th>quotation_id</th>
      <th>resource</th>
      <th>activity</th>
      <th>action</th>
      <th>timestamp</th>
      <th>activity_cost</th>
      <th>case_id</th>
      <th>case:concept:name</th>
      <th>concept:name</th>
      <th>time:timestamp</th>
      <th>@@index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Katherine</td>
      <td>Receive Sales Order</td>
      <td>order-17-56542-0000001</td>
      <td>2017-10-17 12:37:22</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Receive Sales Order</td>
      <td>2017-10-17 12:37:22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Jonathan</td>
      <td>Send To Manufacturing</td>
      <td>order-17-56542-0000002</td>
      <td>2017-10-19 15:30:40</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Send To Manufacturing</td>
      <td>2017-10-19 15:30:40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6048</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Amy</td>
      <td>Order Materials</td>
      <td>order-17-56542-0006049</td>
      <td>2017-10-22 14:02:01</td>
      <td>4344.947965</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Order Materials</td>
      <td>2017-10-22 14:02:01</td>
      <td>6048</td>
    </tr>
    <tr>
      <th>6049</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Julia</td>
      <td>Receive Materials</td>
      <td>order-17-56542-0006050</td>
      <td>2017-10-25 11:30:24</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Receive Materials</td>
      <td>2017-10-25 11:30:24</td>
      <td>6049</td>
    </tr>
    <tr>
      <th>15144</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Gregory</td>
      <td>Schedule Job</td>
      <td>order-17-56542-0015145</td>
      <td>2017-10-27 11:32:05</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Schedule Job</td>
      <td>2017-10-27 11:32:05</td>
      <td>15144</td>
    </tr>
    <tr>
      <th>15145</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Jerry</td>
      <td>Produce Order</td>
      <td>order-17-56542-0015146</td>
      <td>2017-11-03 18:34:00</td>
      <td>25810.513393</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Produce Order</td>
      <td>2017-11-03 18:34:00</td>
      <td>15145</td>
    </tr>
    <tr>
      <th>26139</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Jerry</td>
      <td>Produce Order</td>
      <td>order-17-56542-0026140</td>
      <td>2017-11-13 11:09:17</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Produce Order</td>
      <td>2017-11-13 11:09:17</td>
      <td>26139</td>
    </tr>
    <tr>
      <th>15146</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Mary</td>
      <td>Quality Control</td>
      <td>order-17-56542-0015147</td>
      <td>2017-11-15 13:43:24</td>
      <td>3156.640994</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Quality Control</td>
      <td>2017-11-15 13:43:24</td>
      <td>15146</td>
    </tr>
    <tr>
      <th>26140</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Mary</td>
      <td>Quality Control</td>
      <td>order-17-56542-0026141</td>
      <td>2017-11-17 12:47:42</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Quality Control</td>
      <td>2017-11-17 12:47:42</td>
      <td>26140</td>
    </tr>
    <tr>
      <th>34110</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Phillip</td>
      <td>Packaging</td>
      <td>order-17-56542-0034111</td>
      <td>2017-11-19 13:48:13</td>
      <td>388.394137</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Packaging</td>
      <td>2017-11-19 13:48:13</td>
      <td>34110</td>
    </tr>
    <tr>
      <th>44359</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Phillip</td>
      <td>Packaging</td>
      <td>order-17-56542-0044360</td>
      <td>2017-11-22 13:29:27</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Packaging</td>
      <td>2017-11-22 13:29:27</td>
      <td>44359</td>
    </tr>
    <tr>
      <th>34111</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Phillip</td>
      <td>Handover To Deliverer</td>
      <td>order-17-56542-0034112</td>
      <td>2017-11-24 18:18:27</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Handover To Deliverer</td>
      <td>2017-11-24 18:18:27</td>
      <td>34111</td>
    </tr>
    <tr>
      <th>47383</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Virginia</td>
      <td>Prepare Invoice</td>
      <td>order-17-56542-0047384</td>
      <td>2017-11-27 15:50:47</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Prepare Invoice</td>
      <td>2017-11-27 15:50:47</td>
      <td>47383</td>
    </tr>
    <tr>
      <th>57780</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Virginia</td>
      <td>Prepare Invoice</td>
      <td>order-17-56542-0057781</td>
      <td>2017-11-29 12:06:34</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Prepare Invoice</td>
      <td>2017-11-29 12:06:34</td>
      <td>57780</td>
    </tr>
    <tr>
      <th>47384</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Virginia</td>
      <td>Send Invoice</td>
      <td>order-17-56542-0047385</td>
      <td>2017-12-01 09:17:18</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Send Invoice</td>
      <td>2017-12-01 09:17:18</td>
      <td>47384</td>
    </tr>
    <tr>
      <th>34112</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Stephen</td>
      <td>Order Delivered</td>
      <td>order-17-56542-0034113</td>
      <td>2017-12-05 14:38:25</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Order Delivered</td>
      <td>2017-12-05 14:38:25</td>
      <td>34112</td>
    </tr>
    <tr>
      <th>47385</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Virginia</td>
      <td>Payment Received</td>
      <td>order-17-56542-0047386</td>
      <td>2017-12-07 09:43:37</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542</td>
      <td>Payment Received</td>
      <td>2017-12-07 09:43:37</td>
      <td>47385</td>
    </tr>
  </tbody>
</table>
</div>




```python
# filter df
filtered_otc_event_log = pm4py.filter_between(otc_event_log_only_so2, "Send Invoice", "Payment Received")
filtered_otc_event_log.shape
```




    (8122, 12)



After filtering with filter_between only those activities of `order-17-56542` remain that belong to the payment subprocess.


```python
# after filter
filtered_otc_event_log = pm4py.filter_between(otc_event_log_only_so2, "Send Invoice", "Payment Received")
filtered_otc_event_log.sort_values(by=['sales_order_id', 'timestamp']).head()
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
      <th>sales_order_id</th>
      <th>quotation_id</th>
      <th>resource</th>
      <th>activity</th>
      <th>action</th>
      <th>timestamp</th>
      <th>activity_cost</th>
      <th>case_id</th>
      <th>case:concept:name</th>
      <th>concept:name</th>
      <th>time:timestamp</th>
      <th>@@index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47384</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Virginia</td>
      <td>Send Invoice</td>
      <td>order-17-56542-0047385</td>
      <td>2017-12-01 09:17:18</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542##@@0</td>
      <td>Send Invoice</td>
      <td>2017-12-01 09:17:18</td>
      <td>47384</td>
    </tr>
    <tr>
      <th>34112</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Stephen</td>
      <td>Order Delivered</td>
      <td>order-17-56542-0034113</td>
      <td>2017-12-05 14:38:25</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542##@@0</td>
      <td>Order Delivered</td>
      <td>2017-12-05 14:38:25</td>
      <td>34112</td>
    </tr>
    <tr>
      <th>47385</th>
      <td>order-17-56542</td>
      <td>NaN</td>
      <td>Virginia</td>
      <td>Payment Received</td>
      <td>order-17-56542-0047386</td>
      <td>2017-12-07 09:43:37</td>
      <td>NaN</td>
      <td>-order-17-56542</td>
      <td>order-17-56542##@@0</td>
      <td>Payment Received</td>
      <td>2017-12-07 09:43:37</td>
      <td>47385</td>
    </tr>
    <tr>
      <th>47387</th>
      <td>order-17-56543</td>
      <td>NaN</td>
      <td>George</td>
      <td>Send Invoice</td>
      <td>order-17-56543-0047388</td>
      <td>2018-01-01 10:18:15</td>
      <td>NaN</td>
      <td>-order-17-56543</td>
      <td>order-17-56543##@@0</td>
      <td>Send Invoice</td>
      <td>2018-01-01 10:18:15</td>
      <td>47387</td>
    </tr>
    <tr>
      <th>34115</th>
      <td>order-17-56543</td>
      <td>NaN</td>
      <td>Cheryl</td>
      <td>Order Fetched</td>
      <td>order-17-56543-0034116</td>
      <td>2018-01-03 13:45:12</td>
      <td>NaN</td>
      <td>-order-17-56543</td>
      <td>order-17-56543##@@0</td>
      <td>Order Fetched</td>
      <td>2018-01-03 13:45:12</td>
      <td>34115</td>
    </tr>
  </tbody>
</table>
</div>




```python
parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'} # identify the case_id_key name (if not change it will simply be the mane of the coloumn)
event_log__otc = log_converter.apply(filtered_otc_event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
```


```python
event_log__otc
```




    [{'attributes': {'concept:name': 'order-17-56542##@@0'}, 'events': [{'sales_order_id': 'order-17-56542', 'quotation_id': nan, 'resource': 'Virginia', 'activity': 'Send Invoice', 'action': 'order-17-56542-0047385', 'timestamp': Timestamp('2017-12-01 09:17:18'), 'activity_cost': nan, 'case_id': '-order-17-56542', 'concept:name': 'Send Invoice', 'time:timestamp': Timestamp('2017-12-01 09:17:18'), '@@index': 47384}, '..', {'sales_order_id': 'order-17-56542', 'quotation_id': nan, 'resource': 'Virginia', 'activity': 'Payment Received', 'action': 'order-17-56542-0047386', 'timestamp': Timestamp('2017-12-07 09:43:37'), 'activity_cost': nan, 'case_id': '-order-17-56542', 'concept:name': 'Payment Received', 'time:timestamp': Timestamp('2017-12-07 09:43:37'), '@@index': 47385}]}, '....', {'attributes': {'concept:name': 'order-17-59565##@@0'}, 'events': [{'sales_order_id': 'order-17-59565', 'quotation_id': nan, 'resource': 'George', 'activity': 'Send Invoice', 'action': 'order-17-59565-0057776', 'timestamp': Timestamp('2017-05-16 19:34:52'), 'activity_cost': nan, 'case_id': '-order-17-59565', 'concept:name': 'Send Invoice', 'time:timestamp': Timestamp('2017-05-16 19:34:52'), '@@index': 57775}, '..', {'sales_order_id': 'order-17-59565', 'quotation_id': nan, 'resource': 'George', 'activity': 'Payment Received', 'action': 'order-17-59565-0057780', 'timestamp': Timestamp('2017-06-08 12:14:44'), 'activity_cost': nan, 'case_id': '-order-17-59565', 'concept:name': 'Payment Received', 'time:timestamp': Timestamp('2017-06-08 12:14:44'), '@@index': 57779}]}]




```python
from pm4py.statistics.traces.generic.log import case_statistics

all_case_durations = case_statistics.get_all_case_durations(event_log__otc, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"}
                                                           )

```


```python
all_case_durations_days = pd.Series(all_case_durations)/86400 # days
```


```python
from matplotlib.pyplot import figure

figure(figsize=(6, 9), dpi=80)
sns.boxplot(data=all_case_durations_days)
plt.title("Distribution of case durations of subprocess 'Payment'")
plt.ylabel("case duration in days")
plt.xlabel("")
plt.show()
```


    
![png](case-study-process-mining-process-analytics-order-to-cash-process-blog_files/case-study-process-mining-process-analytics-order-to-cash-process-blog_66_0.png)
    



```python
# it seems no title can be parametrized through view_dotted_chart method
pm4py.view_dotted_chart(event_log__otc)
```


    
![png](case-study-process-mining-process-analytics-order-to-cash-process-blog_files/case-study-process-mining-process-analytics-order-to-cash-process-blog_67_0.png)
    


Having only limited time to dig in, I managed to do quite a lot of types of analyses with pm4py, however it is not that widely used and the documentation of its methods are bare bone basic. Often essential requirements in order to make certain methods work are not communicated unfortunately. For example there seems to be this requirement hat the case id column has to be the first column in Logdata DataFrame, so that the `between filter` actually returns what was intended to be returned. While searching for these kind of oddities certainly sucks, I will follow pm4py progress. Cheers!

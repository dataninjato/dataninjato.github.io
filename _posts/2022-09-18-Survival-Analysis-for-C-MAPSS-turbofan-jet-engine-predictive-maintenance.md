---
title: Survival Analysis for C-MAPSS turbofan jet engines ~ predictive maintenance
tags: [Survival Analysis, Python, Time Series, predictive maintenance]
style: fill
color: danger
description: The NASA C-MAPSS multivariate timeseries data set is about a set of identical turbofan jet engines all showing different degrees of initial wear and manufacturing variations as well as operational settings. We apply statistical methods of Survival Analysis to analyze the length of time until the undesired occurrence of an failure event can be observed.
---

Data Set FD002 of the NASA C-MAPSS multivariate timeseries data set is about a set of identical turbofan jet engines all showing different degrees of initial wear and manufacturing variations as well as operational settings.
[Data Set Information](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

This project is about applying the statistical methods of **Survival Analysis** to analyze the length of time required until the occurrence of an event. Survival Analysis is often referred to as **time-to-event analysis**. Through this survival analysis method, we can plan well or take steps to anticipate an unwanted event. Survival analysis was first developed in the field of medicine to see the effect of a medical treatment on the patient's survival until death. In addition to the application in the field of medicine, this method can be applied to the following fields or industries:

1. **Manufacturing**: Predicting the duration of a machine running until it breaks down.
2. **Marketing**: Predict the duration of a customer's subscription to unsubscribe.
3. **Human Resources**: Predict the duration of an employee's employment until resignation.

In Survival Analysis we may observe entities of interest that started a process or came into existance before we start our data gathering or in contrast have not yet finished the process or at least the anticipated time to event has not passed at the time of our end of observation. Hence these censoring cases can be distinguished:

* **Not censored**:
    1. If the event occurred **AND**
    2. the survival duration is known.
    

* **Right censored**: (most common type)
    1. If the event did not occur in observation time **OR**
    2. the true duration is longer than our observation.
    
    
* **Left censored**: 
    1. If the event has happened **AND** 
    2. we know the true duration is shorter than our observation time.
    

* **Interval censored**:
    1. The event is observed, but individuals come in and out of observation, 
    so the exact event times and survival durations are unknown, e.g. due to late discovery in periodical maintenance

These definitions are often misunderstood especially in context of graphical timeline representations, hence I kept the conditions above exact and concise and omitted visualization on purpose.

Due to *censored observations* the use of ordinary regression models is not an option. We will use survival analysis methods which can be used with censored data.

### Mathematical representation

Suppose $T$ is a random variable that expresses the time until the occurrence of an event. Then

1. **Probability Density Function (PDF)** $f(t) = P(T = t)$: the probability of an event occurring at time $t$
2. **Cumulative Distribution Function* (CDF)** $F(t) = P(T <= t)$: the probability of an event occurring before time $t$. Mathematically, CDF can be defined as follows:
$$
F(t) = \int_{0}^{t}{f(s)} ds
$$
3. **Survival Distribution Function* (SDF)** $S(t) = P(T>t)$: the probability of an event occurring after time $t$. Mathematically, the SDF can be defined as follows:
$ S(t) = 1 - F(t) $
4. **Hazard Function (*Hazard Rate Function/Force of Mortality*)** $h(t)$: the conditional probability of an event occurring at time $t$ knowing the observed subject has not experienced the event at time $t$. Mathematically, the hazard function can be obtained as follows: $ h(t) = - \frac{d}{dt} ln(S(t))$



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set(rc = {'figure.figsize':(16,9)})

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

## Import Data and Preprocessing

We are going to use the [NASA turbofan jet engine data set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) which contains rows as time series data across the time cycles that the engines have performed. In each of these cycles, three types of operational settings were recorded as wells as 22 different measurements through sensors.


```python
columns = ["machine_name", "cycle", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + [f'sensor_measurement_{i:02}' for i in range(1,22)]
```


```python
# Read data
turbofan_df = pd.read_csv("train_FD002.txt", header = None, sep = "\s+", names = columns)
turbofan_df_test = pd.read_csv("test_FD002.txt", header = None, sep = "\s+", names = columns)
turbofan_df.head()
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>...</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>34.9983</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>555.32</td>
      <td>1358.61</td>
      <td>1137.23</td>
      <td>5.48</td>
      <td>...</td>
      <td>183.06</td>
      <td>2387.72</td>
      <td>8048.56</td>
      <td>9.3461</td>
      <td>0.02</td>
      <td>334</td>
      <td>2223</td>
      <td>100.00</td>
      <td>14.73</td>
      <td>8.8071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>41.9982</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.90</td>
      <td>1353.22</td>
      <td>1125.78</td>
      <td>3.91</td>
      <td>...</td>
      <td>130.42</td>
      <td>2387.66</td>
      <td>8072.30</td>
      <td>9.3774</td>
      <td>0.02</td>
      <td>330</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.41</td>
      <td>6.2665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>24.9988</td>
      <td>0.6218</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.31</td>
      <td>1256.76</td>
      <td>1047.45</td>
      <td>7.05</td>
      <td>...</td>
      <td>164.22</td>
      <td>2028.03</td>
      <td>7864.87</td>
      <td>10.8941</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.08</td>
      <td>8.6723</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>42.0077</td>
      <td>0.8416</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.51</td>
      <td>1354.03</td>
      <td>1126.38</td>
      <td>3.91</td>
      <td>...</td>
      <td>130.72</td>
      <td>2387.61</td>
      <td>8068.66</td>
      <td>9.3528</td>
      <td>0.02</td>
      <td>329</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.59</td>
      <td>6.4701</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>25.0005</td>
      <td>0.6203</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.07</td>
      <td>1257.71</td>
      <td>1047.93</td>
      <td>7.05</td>
      <td>...</td>
      <td>164.31</td>
      <td>2028.00</td>
      <td>7861.23</td>
      <td>10.8963</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.13</td>
      <td>8.5286</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
turbofan_df[turbofan_df.machine_name == 9
           ]
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>...</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1513</th>
      <td>9</td>
      <td>1</td>
      <td>35.0044</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>555.34</td>
      <td>1353.35</td>
      <td>1127.38</td>
      <td>5.48</td>
      <td>...</td>
      <td>182.78</td>
      <td>2387.88</td>
      <td>8058.44</td>
      <td>9.3193</td>
      <td>0.02</td>
      <td>334</td>
      <td>2223</td>
      <td>100.0</td>
      <td>15.04</td>
      <td>8.8947</td>
    </tr>
    <tr>
      <th>1514</th>
      <td>9</td>
      <td>2</td>
      <td>10.0029</td>
      <td>0.2500</td>
      <td>100.0</td>
      <td>489.05</td>
      <td>604.82</td>
      <td>1494.69</td>
      <td>1311.21</td>
      <td>10.52</td>
      <td>...</td>
      <td>372.01</td>
      <td>2388.15</td>
      <td>8120.88</td>
      <td>8.6418</td>
      <td>0.03</td>
      <td>370</td>
      <td>2319</td>
      <td>100.0</td>
      <td>28.31</td>
      <td>17.1242</td>
    </tr>
    <tr>
      <th>1515</th>
      <td>9</td>
      <td>3</td>
      <td>10.0012</td>
      <td>0.2508</td>
      <td>100.0</td>
      <td>489.05</td>
      <td>604.39</td>
      <td>1494.88</td>
      <td>1304.80</td>
      <td>10.52</td>
      <td>...</td>
      <td>371.81</td>
      <td>2388.17</td>
      <td>8120.16</td>
      <td>8.6273</td>
      <td>0.03</td>
      <td>369</td>
      <td>2319</td>
      <td>100.0</td>
      <td>28.59</td>
      <td>17.0821</td>
    </tr>
    <tr>
      <th>1516</th>
      <td>9</td>
      <td>4</td>
      <td>10.0046</td>
      <td>0.2505</td>
      <td>100.0</td>
      <td>489.05</td>
      <td>604.52</td>
      <td>1490.63</td>
      <td>1311.28</td>
      <td>10.52</td>
      <td>...</td>
      <td>371.83</td>
      <td>2388.14</td>
      <td>8116.98</td>
      <td>8.6594</td>
      <td>0.03</td>
      <td>369</td>
      <td>2319</td>
      <td>100.0</td>
      <td>28.51</td>
      <td>17.0908</td>
    </tr>
    <tr>
      <th>1517</th>
      <td>9</td>
      <td>5</td>
      <td>0.0014</td>
      <td>0.0006</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.30</td>
      <td>1586.92</td>
      <td>1412.84</td>
      <td>14.62</td>
      <td>...</td>
      <td>522.19</td>
      <td>2388.07</td>
      <td>8130.64</td>
      <td>8.4464</td>
      <td>0.03</td>
      <td>391</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.99</td>
      <td>23.4381</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>1707</th>
      <td>9</td>
      <td>195</td>
      <td>42.0064</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.60</td>
      <td>1374.56</td>
      <td>1139.37</td>
      <td>3.91</td>
      <td>...</td>
      <td>130.83</td>
      <td>2387.37</td>
      <td>8066.38</td>
      <td>9.4609</td>
      <td>0.02</td>
      <td>332</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.45</td>
      <td>6.3269</td>
    </tr>
    <tr>
      <th>1708</th>
      <td>9</td>
      <td>196</td>
      <td>10.0023</td>
      <td>0.2500</td>
      <td>100.0</td>
      <td>489.05</td>
      <td>605.97</td>
      <td>1517.57</td>
      <td>1321.09</td>
      <td>10.52</td>
      <td>...</td>
      <td>370.75</td>
      <td>2388.38</td>
      <td>8107.72</td>
      <td>8.7508</td>
      <td>0.03</td>
      <td>373</td>
      <td>2319</td>
      <td>100.0</td>
      <td>28.40</td>
      <td>16.9684</td>
    </tr>
    <tr>
      <th>1709</th>
      <td>9</td>
      <td>197</td>
      <td>42.0077</td>
      <td>0.8402</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>551.00</td>
      <td>1368.98</td>
      <td>1145.31</td>
      <td>3.91</td>
      <td>...</td>
      <td>129.82</td>
      <td>2387.39</td>
      <td>8063.45</td>
      <td>9.5044</td>
      <td>0.02</td>
      <td>333</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.37</td>
      <td>6.3352</td>
    </tr>
    <tr>
      <th>1710</th>
      <td>9</td>
      <td>198</td>
      <td>20.0015</td>
      <td>0.7000</td>
      <td>100.0</td>
      <td>491.19</td>
      <td>608.62</td>
      <td>1499.60</td>
      <td>1274.34</td>
      <td>9.35</td>
      <td>...</td>
      <td>313.58</td>
      <td>2388.16</td>
      <td>8040.91</td>
      <td>9.3245</td>
      <td>0.03</td>
      <td>371</td>
      <td>2324</td>
      <td>100.0</td>
      <td>24.16</td>
      <td>14.5259</td>
    </tr>
    <tr>
      <th>1711</th>
      <td>9</td>
      <td>199</td>
      <td>10.0047</td>
      <td>0.2519</td>
      <td>100.0</td>
      <td>489.05</td>
      <td>605.75</td>
      <td>1515.93</td>
      <td>1332.19</td>
      <td>10.52</td>
      <td>...</td>
      <td>369.80</td>
      <td>2388.42</td>
      <td>8102.98</td>
      <td>8.7351</td>
      <td>0.03</td>
      <td>373</td>
      <td>2319</td>
      <td>100.0</td>
      <td>28.34</td>
      <td>16.9781</td>
    </tr>
  </tbody>
</table>
<p>199 rows × 26 columns</p>
</div>



### Create Censored Data

If we look at the data above, the data we have is not data that has *right censored* observations. To get these observations, we do these steps: Select the maximum cycles until machine failure for each machine and assume the end of observation time.


```python
# backup timeseries data
timeseries_df = turbofan_df.copy()
timeseries_df_test = turbofan_df_test.copy()
timeseries_df.head()
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>...</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>34.9983</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>555.32</td>
      <td>1358.61</td>
      <td>1137.23</td>
      <td>5.48</td>
      <td>...</td>
      <td>183.06</td>
      <td>2387.72</td>
      <td>8048.56</td>
      <td>9.3461</td>
      <td>0.02</td>
      <td>334</td>
      <td>2223</td>
      <td>100.00</td>
      <td>14.73</td>
      <td>8.8071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>41.9982</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.90</td>
      <td>1353.22</td>
      <td>1125.78</td>
      <td>3.91</td>
      <td>...</td>
      <td>130.42</td>
      <td>2387.66</td>
      <td>8072.30</td>
      <td>9.3774</td>
      <td>0.02</td>
      <td>330</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.41</td>
      <td>6.2665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>24.9988</td>
      <td>0.6218</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.31</td>
      <td>1256.76</td>
      <td>1047.45</td>
      <td>7.05</td>
      <td>...</td>
      <td>164.22</td>
      <td>2028.03</td>
      <td>7864.87</td>
      <td>10.8941</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.08</td>
      <td>8.6723</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>42.0077</td>
      <td>0.8416</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.51</td>
      <td>1354.03</td>
      <td>1126.38</td>
      <td>3.91</td>
      <td>...</td>
      <td>130.72</td>
      <td>2387.61</td>
      <td>8068.66</td>
      <td>9.3528</td>
      <td>0.02</td>
      <td>329</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.59</td>
      <td>6.4701</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>25.0005</td>
      <td>0.6203</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.07</td>
      <td>1257.71</td>
      <td>1047.93</td>
      <td>7.05</td>
      <td>...</td>
      <td>164.31</td>
      <td>2028.00</td>
      <td>7861.23</td>
      <td>10.8963</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.13</td>
      <td>8.5286</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Select maximum cycle
max_cycle = turbofan_df.groupby(by = "machine_name")['cycle'].transform(max)
#max_cycle = turbofan_df.groupby(by = "machine_name")['cycle'].max()
max_cycle
```




    0        149
    1        149
    2        149
    3        149
    4        149
            ... 
    53754    316
    53755    316
    53756    316
    53757    316
    53758    316
    Name: cycle, Length: 53759, dtype: int64




```python
turbofan_df = turbofan_df[turbofan_df["cycle"] == max_cycle].set_index('machine_name')
turbofan_df.head()
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
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>sensor_measurement_06</th>
      <th>...</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
    </tr>
    <tr>
      <th>machine_name</th>
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
      <th>1</th>
      <td>149</td>
      <td>42.0017</td>
      <td>0.8414</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.49</td>
      <td>1366.01</td>
      <td>1149.81</td>
      <td>3.91</td>
      <td>5.71</td>
      <td>...</td>
      <td>129.55</td>
      <td>2387.40</td>
      <td>8066.19</td>
      <td>9.4765</td>
      <td>0.02</td>
      <td>332</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.45</td>
      <td>6.2285</td>
    </tr>
    <tr>
      <th>2</th>
      <td>269</td>
      <td>42.0047</td>
      <td>0.8411</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.11</td>
      <td>1368.75</td>
      <td>1146.65</td>
      <td>3.91</td>
      <td>5.72</td>
      <td>...</td>
      <td>129.76</td>
      <td>2388.42</td>
      <td>8110.26</td>
      <td>9.4315</td>
      <td>0.02</td>
      <td>334</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.56</td>
      <td>6.2615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>206</td>
      <td>42.0073</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.80</td>
      <td>1356.97</td>
      <td>1144.89</td>
      <td>3.91</td>
      <td>5.72</td>
      <td>...</td>
      <td>130.02</td>
      <td>2387.87</td>
      <td>8082.25</td>
      <td>9.4962</td>
      <td>0.02</td>
      <td>333</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.46</td>
      <td>6.3349</td>
    </tr>
    <tr>
      <th>4</th>
      <td>235</td>
      <td>0.0030</td>
      <td>0.0007</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>643.68</td>
      <td>1605.86</td>
      <td>1428.21</td>
      <td>14.62</td>
      <td>21.61</td>
      <td>...</td>
      <td>520.25</td>
      <td>2388.17</td>
      <td>8215.14</td>
      <td>8.5784</td>
      <td>0.03</td>
      <td>397</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.47</td>
      <td>22.9717</td>
    </tr>
    <tr>
      <th>5</th>
      <td>154</td>
      <td>42.0049</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.53</td>
      <td>1364.82</td>
      <td>1146.87</td>
      <td>3.91</td>
      <td>5.72</td>
      <td>...</td>
      <td>130.05</td>
      <td>2389.19</td>
      <td>8151.36</td>
      <td>9.4339</td>
      <td>0.02</td>
      <td>333</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.74</td>
      <td>6.3906</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



Next, we make a lollipop plot to obtain the distribution of maximum cycles on each machine.


```python
# Lollipop plot for each machine name
plt.hlines(y=turbofan_df.index, xmin=1, xmax=turbofan_df['cycle'], color='darkgrey')
plt.plot(turbofan_df['cycle'], turbofan_df.index, "o", color='royalblue')
plt.plot([1 for i in range(len(turbofan_df))], turbofan_df.index, "o")

ylim = plt.gca().get_ylim()
plt.ylim(ylim[1], ylim[0])

# Add titles and axis names
plt.title("Cycle runtime of engine until failure event")
plt.xlabel("Cycle")
plt.ylabel('Machine ID')

# Show the plot
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_17_0.png)
    


> We assume that the observation time limit is 220 cycles so that when the machine is still active after 220 cycles then the machine will be considered right censored.


```python
# Create status column
turbofan_df['status'] = turbofan_df['cycle'].apply(lambda x: False if x > 215 else True)
```


```python
turbofan_df.head(5)
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
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>sensor_measurement_06</th>
      <th>...</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
      <th>status</th>
    </tr>
    <tr>
      <th>machine_name</th>
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
      <th>1</th>
      <td>149</td>
      <td>42.0017</td>
      <td>0.8414</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.49</td>
      <td>1366.01</td>
      <td>1149.81</td>
      <td>3.91</td>
      <td>5.71</td>
      <td>...</td>
      <td>2387.40</td>
      <td>8066.19</td>
      <td>9.4765</td>
      <td>0.02</td>
      <td>332</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.45</td>
      <td>6.2285</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>269</td>
      <td>42.0047</td>
      <td>0.8411</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.11</td>
      <td>1368.75</td>
      <td>1146.65</td>
      <td>3.91</td>
      <td>5.72</td>
      <td>...</td>
      <td>2388.42</td>
      <td>8110.26</td>
      <td>9.4315</td>
      <td>0.02</td>
      <td>334</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.56</td>
      <td>6.2615</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>206</td>
      <td>42.0073</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.80</td>
      <td>1356.97</td>
      <td>1144.89</td>
      <td>3.91</td>
      <td>5.72</td>
      <td>...</td>
      <td>2387.87</td>
      <td>8082.25</td>
      <td>9.4962</td>
      <td>0.02</td>
      <td>333</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.46</td>
      <td>6.3349</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>235</td>
      <td>0.0030</td>
      <td>0.0007</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>643.68</td>
      <td>1605.86</td>
      <td>1428.21</td>
      <td>14.62</td>
      <td>21.61</td>
      <td>...</td>
      <td>2388.17</td>
      <td>8215.14</td>
      <td>8.5784</td>
      <td>0.03</td>
      <td>397</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.47</td>
      <td>22.9717</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>154</td>
      <td>42.0049</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.53</td>
      <td>1364.82</td>
      <td>1146.87</td>
      <td>3.91</td>
      <td>5.72</td>
      <td>...</td>
      <td>2389.19</td>
      <td>8151.36</td>
      <td>9.4339</td>
      <td>0.02</td>
      <td>333</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.74</td>
      <td>6.3906</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



> Machine status `True` indicates that the machine has failed within the observation time span while `False` indicates the machine has not failed yet during the observation time span.


```python
# Distribution of each status
turbofan_df['status'].value_counts()
```




    True     173
    False     87
    Name: status, dtype: int64



## Exploratory Data Analysis

The next step is to perform *feature selection*, which is the selection of variables as predictors that we want to include in the model.


```python
turbofan_df.describe()
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
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>sensor_measurement_06</th>
      <th>sensor_measurement_07</th>
      <th>...</th>
      <th>sensor_measurement_11</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>...</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
      <td>260.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>206.765385</td>
      <td>27.937113</td>
      <td>0.598383</td>
      <td>466.689346</td>
      <td>575.619115</td>
      <td>1425.801000</td>
      <td>1218.674692</td>
      <td>7.124231</td>
      <td>10.386385</td>
      <td>255.637115</td>
      <td>...</td>
      <td>43.836038</td>
      <td>240.824538</td>
      <td>2364.920731</td>
      <td>8105.652538</td>
      <td>9.288340</td>
      <td>349.815385</td>
      <td>2242.073077</td>
      <td>99.014654</td>
      <td>18.644115</td>
      <td>11.185072</td>
    </tr>
    <tr>
      <th>std</th>
      <td>46.782198</td>
      <td>16.564435</td>
      <td>0.324420</td>
      <td>27.479073</td>
      <td>36.085540</td>
      <td>97.552983</td>
      <td>114.934975</td>
      <td>3.949434</td>
      <td>5.864195</td>
      <td>153.443600</td>
      <td>...</td>
      <td>2.682455</td>
      <td>144.687933</td>
      <td>89.267941</td>
      <td>74.714146</td>
      <td>0.584163</td>
      <td>25.629762</td>
      <td>108.889934</td>
      <td>3.732538</td>
      <td>10.443101</td>
      <td>6.261529</td>
    </tr>
    <tr>
      <th>min</th>
      <td>128.000000</td>
      <td>0.000100</td>
      <td>0.000000</td>
      <td>445.000000</td>
      <td>536.620000</td>
      <td>1263.290000</td>
      <td>1058.960000</td>
      <td>3.910000</td>
      <td>5.710000</td>
      <td>136.990000</td>
      <td>...</td>
      <td>37.210000</td>
      <td>129.330000</td>
      <td>2027.610000</td>
      <td>7850.960000</td>
      <td>8.486100</td>
      <td>309.000000</td>
      <td>1915.000000</td>
      <td>84.930000</td>
      <td>10.270000</td>
      <td>6.164800</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>174.000000</td>
      <td>10.003225</td>
      <td>0.250650</td>
      <td>445.000000</td>
      <td>550.477500</td>
      <td>1364.650000</td>
      <td>1143.535000</td>
      <td>3.910000</td>
      <td>5.720000</td>
      <td>138.277500</td>
      <td>...</td>
      <td>42.650000</td>
      <td>130.280000</td>
      <td>2387.970000</td>
      <td>8081.580000</td>
      <td>8.753125</td>
      <td>334.000000</td>
      <td>2212.000000</td>
      <td>100.000000</td>
      <td>10.470000</td>
      <td>6.283750</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>199.000000</td>
      <td>41.998300</td>
      <td>0.840000</td>
      <td>445.000000</td>
      <td>550.880000</td>
      <td>1370.270000</td>
      <td>1147.880000</td>
      <td>3.910000</td>
      <td>5.720000</td>
      <td>139.165000</td>
      <td>...</td>
      <td>42.770000</td>
      <td>130.985000</td>
      <td>2388.320000</td>
      <td>8116.755000</td>
      <td>9.414700</td>
      <td>335.000000</td>
      <td>2212.000000</td>
      <td>100.000000</td>
      <td>10.685000</td>
      <td>6.390250</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>230.250000</td>
      <td>42.002625</td>
      <td>0.840100</td>
      <td>489.050000</td>
      <td>606.042500</td>
      <td>1512.122500</td>
      <td>1329.327500</td>
      <td>10.520000</td>
      <td>15.500000</td>
      <td>392.595000</td>
      <td>...</td>
      <td>46.070000</td>
      <td>370.152500</td>
      <td>2388.752500</td>
      <td>8148.730000</td>
      <td>9.456725</td>
      <td>372.000000</td>
      <td>2319.000000</td>
      <td>100.000000</td>
      <td>28.210000</td>
      <td>16.925775</td>
    </tr>
    <tr>
      <th>max</th>
      <td>378.000000</td>
      <td>42.007900</td>
      <td>0.842000</td>
      <td>518.670000</td>
      <td>644.260000</td>
      <td>1609.460000</td>
      <td>1438.160000</td>
      <td>14.620000</td>
      <td>21.610000</td>
      <td>551.980000</td>
      <td>...</td>
      <td>48.510000</td>
      <td>520.250000</td>
      <td>2390.390000</td>
      <td>8252.610000</td>
      <td>11.045400</td>
      <td>398.000000</td>
      <td>2388.000000</td>
      <td>100.000000</td>
      <td>38.680000</td>
      <td>23.153900</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>




```python
turbofan_df.iloc[:,8:].head()
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
      <th>sensor_measurement_05</th>
      <th>sensor_measurement_06</th>
      <th>sensor_measurement_07</th>
      <th>sensor_measurement_08</th>
      <th>sensor_measurement_09</th>
      <th>sensor_measurement_10</th>
      <th>sensor_measurement_11</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
      <th>status</th>
    </tr>
    <tr>
      <th>machine_name</th>
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
      <th>1</th>
      <td>3.91</td>
      <td>5.71</td>
      <td>137.76</td>
      <td>2211.38</td>
      <td>8307.72</td>
      <td>1.02</td>
      <td>42.77</td>
      <td>129.55</td>
      <td>2387.40</td>
      <td>8066.19</td>
      <td>9.4765</td>
      <td>0.02</td>
      <td>332</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.45</td>
      <td>6.2285</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.91</td>
      <td>5.72</td>
      <td>138.61</td>
      <td>2212.32</td>
      <td>8358.80</td>
      <td>1.02</td>
      <td>42.64</td>
      <td>129.76</td>
      <td>2388.42</td>
      <td>8110.26</td>
      <td>9.4315</td>
      <td>0.02</td>
      <td>334</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.56</td>
      <td>6.2615</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.91</td>
      <td>5.72</td>
      <td>138.08</td>
      <td>2211.81</td>
      <td>8329.26</td>
      <td>1.02</td>
      <td>42.85</td>
      <td>130.02</td>
      <td>2387.87</td>
      <td>8082.25</td>
      <td>9.4962</td>
      <td>0.02</td>
      <td>333</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.46</td>
      <td>6.3349</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.62</td>
      <td>21.61</td>
      <td>551.49</td>
      <td>2388.17</td>
      <td>9150.14</td>
      <td>1.30</td>
      <td>48.25</td>
      <td>520.25</td>
      <td>2388.17</td>
      <td>8215.14</td>
      <td>8.5784</td>
      <td>0.03</td>
      <td>397</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.47</td>
      <td>22.9717</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.91</td>
      <td>5.72</td>
      <td>138.48</td>
      <td>2212.99</td>
      <td>8397.52</td>
      <td>1.02</td>
      <td>42.67</td>
      <td>130.05</td>
      <td>2389.19</td>
      <td>8151.36</td>
      <td>9.4339</td>
      <td>0.02</td>
      <td>333</td>
      <td>2212</td>
      <td>100.0</td>
      <td>10.74</td>
      <td>6.3906</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Check Uniqueness

Pertama kita akan memeriksa banyak nilai unik pada setiap kolom lalu kolom yang memiliki nilai unik yang sedikit akan diganti dengan tipe data kategori.


```python
turbofan_df.nunique()
```




    cycle                    133
    operational_setting_1    165
    operational_setting_2     62
    operational_setting_3      2
    sensor_measurement_01      6
    sensor_measurement_02    188
    sensor_measurement_03    257
    sensor_measurement_04    251
    sensor_measurement_05      6
    sensor_measurement_06      8
    sensor_measurement_07    209
    sensor_measurement_08    177
    sensor_measurement_09    259
    sensor_measurement_10      8
    sensor_measurement_11    123
    sensor_measurement_12    198
    sensor_measurement_13    152
    sensor_measurement_14    258
    sensor_measurement_15    247
    sensor_measurement_16      2
    sensor_measurement_17     27
    sensor_measurement_18      6
    sensor_measurement_19      2
    sensor_measurement_20    135
    sensor_measurement_21    253
    status                     2
    dtype: int64



> Columns that can be changed to category type: `operational_setting_3`, `sensor_measurement_16`


```python
# Change to category
cat = ['operational_setting_3', 'sensor_measurement_16']

turbofan_df[cat] = turbofan_df[cat].astype('category')
turbofan_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 260 entries, 1 to 260
    Data columns (total 26 columns):
     #   Column                 Non-Null Count  Dtype   
    ---  ------                 --------------  -----   
     0   cycle                  260 non-null    int64   
     1   operational_setting_1  260 non-null    float64 
     2   operational_setting_2  260 non-null    float64 
     3   operational_setting_3  260 non-null    category
     4   sensor_measurement_01  260 non-null    float64 
     5   sensor_measurement_02  260 non-null    float64 
     6   sensor_measurement_03  260 non-null    float64 
     7   sensor_measurement_04  260 non-null    float64 
     8   sensor_measurement_05  260 non-null    float64 
     9   sensor_measurement_06  260 non-null    float64 
     10  sensor_measurement_07  260 non-null    float64 
     11  sensor_measurement_08  260 non-null    float64 
     12  sensor_measurement_09  260 non-null    float64 
     13  sensor_measurement_10  260 non-null    float64 
     14  sensor_measurement_11  260 non-null    float64 
     15  sensor_measurement_12  260 non-null    float64 
     16  sensor_measurement_13  260 non-null    float64 
     17  sensor_measurement_14  260 non-null    float64 
     18  sensor_measurement_15  260 non-null    float64 
     19  sensor_measurement_16  260 non-null    category
     20  sensor_measurement_17  260 non-null    int64   
     21  sensor_measurement_18  260 non-null    int64   
     22  sensor_measurement_19  260 non-null    float64 
     23  sensor_measurement_20  260 non-null    float64 
     24  sensor_measurement_21  260 non-null    float64 
     25  status                 260 non-null    bool    
    dtypes: bool(1), category(2), float64(20), int64(3)
    memory usage: 57.9 KB
    

### Correlation Heatmap

We can check the correlation between columns of candidate predictors to see if there is a high correlation between predictors. Variables that have a high correlation with other variables need to be selected to avoid multicollinearity.


```python
sns.heatmap(turbofan_df.corr()**2, annot = True,)
```




    <AxesSubplot:>




    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_34_1.png)
    


> In the plot above, the correlation between variables looks quite high. In this case we will try to first select the columns `sensor_measurement_04`, `sensor_measurement_08`, `sensor_measurement_11`, `sensor_measurement_14`.


```python
selected_columns = ['sensor_measurement_04','sensor_measurement_11','sensor_measurement_14']
cleaned_data = turbofan_df.loc[:, selected_columns + cat + ['status', 'cycle']]
sns.heatmap(cleaned_data.corr(), annot = True)
```




    <AxesSubplot:>




    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_36_1.png)
    


## Data Preparation for Modeling

The next step is to prepare the data for modeling. Things that need to be considered are:

1. For columns with category type, One Hot Encoding needs to be done.
2. For the target data, it needs to be written in the form of an array with each element is a tuple consisting of the machine status (True/False) and the time/cycle


```python
# One Hot Encoding for Categorical Variable
from sksurv.preprocessing import OneHotEncoder

data_x = OneHotEncoder().fit_transform(cleaned_data.iloc[:, :-2])
data_x.head()
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
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_11</th>
      <th>sensor_measurement_14</th>
      <th>operational_setting_3=100.0</th>
      <th>sensor_measurement_16=0.03</th>
    </tr>
    <tr>
      <th>machine_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1149.81</td>
      <td>42.77</td>
      <td>8066.19</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1146.65</td>
      <td>42.64</td>
      <td>8110.26</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1144.89</td>
      <td>42.85</td>
      <td>8082.25</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1428.21</td>
      <td>48.25</td>
      <td>8215.14</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1146.87</td>
      <td>42.67</td>
      <td>8151.36</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Preprocessing for target variable

data_y = list(cleaned_data.loc[:, ["status", "cycle"]].itertuples(index = None, name = None))
data_y = np.array(data_y, dtype=[('status', bool), ('cycle', float)])
data_y[:5]
```




    array([( True, 149.), (False, 269.), ( True, 206.), (False, 235.),
           ( True, 154.)], dtype=[('status', '?'), ('cycle', '<f8')])




```python
# Preprocessing for target variable

target_data = list(cleaned_data.loc[:, ["status", "cycle"]].itertuples(index = None, name = None))
target_data = np.array(target_data, dtype=[('status', bool), ('cycle', float)])
target_data[:5]
```




    array([( True, 149.), (False, 269.), ( True, 206.), (False, 235.),
           ( True, 154.)], dtype=[('status', '?'), ('cycle', '<f8')])



## Kaplan-Meier Estimator

From the data we have above, we can build a survival function **Kaplan Meier Estimator**. The Kaplan Meier Estimator is built by calculating the probability of observations that survive (do not experience events) at each time. Mathematically, the Kaplan Meier Estimator can be written as follows:
$$
S(t) = \prod_{j=1}^{t} \frac{n_j - d_j}{n_j} 
$$
where $n_j$ is the number of subjects at time $j$ and $d_j$ is the number of subjects that experienced the event.



```python
from lifelines import KaplanMeierFitter

# Instantiate Kaplan Meier object
kmf = KaplanMeierFitter()

# Fit Kaplan Meier estimator
kmf.fit(durations=target_data['cycle'], event_observed=target_data['status'])
```




    <lifelines.KaplanMeierFitter:"KM_estimate", fitted with 260 total observations, 87 right-censored observations>




```python
# Plot survival function with confidence intervals
kmf.plot_survival_function()

plt.title("Kaplan-Meier Estimator")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")

plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_45_0.png)
    


**Interpretation:**

1. Until the time $t=125$ the value of $S(t)=1$ which indicates that all machines are still running until the 125th cycle
2. When $t=175$ the value of $S(t)$ is around 0.75 indicating that after about 75% the engine is still running until the 175th cycle
3. The chance of a engine still having not failed is 50% at cycle 190.

### Compare Survival Time for Each value in categorical columns

We have learned how to construct a survival function using the Kaplan Meier Estimator. Sometimes we want to do a comparison on whether different categories/treatments will affect the survival of our subjects.

For example, in `cleaned_data` we have two columns that have category types `operational_setting_3` and `sensor_measurement_16`. Let's check the number of unique values in these two columns

In this section, we will compare the distribution of survival time for each category in every categorical columns. From the distribution we hope that we can determine if there is a difference distribution for each category or not.


```python
os3_unique = list(cleaned_data['operational_setting_3'].unique())
sm16_unique = list(cleaned_data['sensor_measurement_16'].unique())
print(f'Unique values operational_setting_3: {os3_unique}')
print(f'Unique values sensor_measurement_16: {sm16_unique}')
```

    Unique values operational_setting_3: [100.0, 60.0]
    Unique values sensor_measurement_16: [0.02, 0.03]
    

Now we will compare the distribution of the survival function between the values 100 and 60 in the `operational_setting_3`.


```python
# plt.xlabel callable bugfix
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')

# Mask for operational_setting_3
fullos3 = (cleaned_data['operational_setting_3'] == 100)

kmf = KaplanMeierFitter()
kmf.fit(durations=cleaned_data[fullos3]['cycle'], event_observed=cleaned_data[fullos3]['status'])
kmf.plot_survival_function(ax=ax)

kmf = KaplanMeierFitter()
kmf.fit(durations=cleaned_data[~fullos3]['cycle'], event_observed=cleaned_data[~fullos3]['status'])
kmf.plot_survival_function(ax=ax)


plt.title(f"Kaplan-Meier Estimator for both operational settings 3")
plt.legend(loc="best")
plt.xlabel('cycles')
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_51_0.png)
    


Please notice that leaving confidence intervals (shaded areas) left aside, there is a broad gap between the survival curves.
With the confidence intervals in the picture however, the overlapping confidence interval areas show that the observed gap might not indicate real differences in the survival functions. There is no significant difference that could be determined.

A different picture is that of two groups separated by a threshold of sensor measurements no 14 where there are significant different survival function curves:


```python
# plt.xlabel callable bugfix
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)

threshold = 8140

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')

# Mask
sm14 = (cleaned_data['sensor_measurement_14'] > threshold)

kmf_exp = KaplanMeierFitter()
kmf_exp.fit(durations=cleaned_data[sm14]['cycle'], event_observed=cleaned_data[sm14]['status'], label='exp')
kmf_exp.plot_survival_function(ax=ax, show_censors=True, ci_legend=True, 
                               label='sm14 greater than ' + str(threshold)
                              )

kmf_control = KaplanMeierFitter()
kmf_control.fit(durations=cleaned_data[~sm14]['cycle'], event_observed=cleaned_data[~sm14]['status'], label='control')
kmf_control.plot_survival_function(ax=ax, show_censors=True, ci_legend=True, 
                                   label='sm14 not greater than ' + str(threshold)
                                  )


from lifelines.plotting import add_at_risk_counts
add_at_risk_counts(kmf_exp, kmf_control, ax=ax)

plt.title(f"Kaplan-Meier Estimator for each Category of Sensor Measurement 14")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel('cycles')
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_53_0.png)
    


### Log-Rank Test

In the previous section, we compared the distribution of survival functions by visualization. However, sometimes further statistical tests are required to determine whether the two distributions can be said to be the same or significantly different. To perform these tests, we can use the **Log-.
Rank Test** with the usual statistical hypotheses.

$H_{0}$ : The distribution of the two survival functions is the same

$H_{1}$ : The distribution of the two survival functions is different


```python
from sksurv.compare import compare_survival

cat.append('sensor_measurement_14')

p_value_list = []
for column in cat:
    p_value = compare_survival(data_y, turbofan_df[column])[1]
    p_value_list.append(p_value)
    
result = pd.DataFrame({'columns': cat, 'p-value': p_value_list}).set_index('columns')
result['conclusion'] = result['p-value'].apply(lambda x: "significant" if x < 0.05 else "not significant")
result
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
      <th>p-value</th>
      <th>conclusion</th>
    </tr>
    <tr>
      <th>columns</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>operational_setting_3</th>
      <td>5.391624e-03</td>
      <td>significant</td>
    </tr>
    <tr>
      <th>sensor_measurement_16</th>
      <td>7.368610e-02</td>
      <td>not significant</td>
    </tr>
    <tr>
      <th>sensor_measurement_14</th>
      <td>4.966909e-145</td>
      <td>significant</td>
    </tr>
  </tbody>
</table>
</div>




```python
cleaned_data
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
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_11</th>
      <th>sensor_measurement_14</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_16</th>
      <th>status</th>
      <th>cycle</th>
    </tr>
    <tr>
      <th>machine_name</th>
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
      <th>1</th>
      <td>1149.81</td>
      <td>42.77</td>
      <td>8066.19</td>
      <td>100.0</td>
      <td>0.02</td>
      <td>True</td>
      <td>149</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1146.65</td>
      <td>42.64</td>
      <td>8110.26</td>
      <td>100.0</td>
      <td>0.02</td>
      <td>False</td>
      <td>269</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1144.89</td>
      <td>42.85</td>
      <td>8082.25</td>
      <td>100.0</td>
      <td>0.02</td>
      <td>True</td>
      <td>206</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1428.21</td>
      <td>48.25</td>
      <td>8215.14</td>
      <td>100.0</td>
      <td>0.03</td>
      <td>False</td>
      <td>235</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1146.87</td>
      <td>42.67</td>
      <td>8151.36</td>
      <td>100.0</td>
      <td>0.02</td>
      <td>True</td>
      <td>154</td>
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
      <th>256</th>
      <td>1425.84</td>
      <td>48.36</td>
      <td>8128.22</td>
      <td>100.0</td>
      <td>0.03</td>
      <td>True</td>
      <td>163</td>
    </tr>
    <tr>
      <th>257</th>
      <td>1146.53</td>
      <td>42.70</td>
      <td>8171.37</td>
      <td>100.0</td>
      <td>0.02</td>
      <td>False</td>
      <td>309</td>
    </tr>
    <tr>
      <th>258</th>
      <td>1326.79</td>
      <td>46.08</td>
      <td>8124.16</td>
      <td>100.0</td>
      <td>0.03</td>
      <td>True</td>
      <td>143</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1149.97</td>
      <td>42.73</td>
      <td>8075.91</td>
      <td>100.0</td>
      <td>0.02</td>
      <td>True</td>
      <td>205</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1145.52</td>
      <td>42.50</td>
      <td>8185.35</td>
      <td>100.0</td>
      <td>0.02</td>
      <td>False</td>
      <td>316</td>
    </tr>
  </tbody>
</table>
<p>260 rows × 7 columns</p>
</div>




```python
cleaned_data[sm14]['cycle'].count()
```




    83




```python
cleaned_data[~sm14]['cycle'].count()
```




    177




```python
# Import logrank_test
from lifelines.statistics import logrank_test

sm16 = (cleaned_data['sensor_measurement_16'] == 0.02)

# Run log-rank test to compare groups by survival function
mask_dict = {'sensor_measurement_14': sm14, 'sensor_measurement_16': sm16, 'operational_setting_3': fullos3}
p_value_list = []

for item in mask_dict.items():
    lrt = logrank_test(durations_A = cleaned_data[item[1]]['cycle'], 
                       durations_B = cleaned_data[~item[1]]['cycle'], 
                       event_observed_A = cleaned_data[item[1]]['status'], 
                       event_observed_B = cleaned_data[~item[1]]['status'])
    p_value_list.append(lrt.p_value)

test_results = pd.DataFrame({'feature': mask_dict.keys(), 'p-value': p_value_list}).set_index('feature')
test_results['conclusion'] = test_results['p-value'].apply(lambda x: "significant" if x < 0.05 else "not significant")
test_results
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
      <th>p-value</th>
      <th>conclusion</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sensor_measurement_14</th>
      <td>5.276097e-07</td>
      <td>significant</td>
    </tr>
    <tr>
      <th>sensor_measurement_16</th>
      <td>7.368610e-02</td>
      <td>not significant</td>
    </tr>
    <tr>
      <th>operational_setting_3</th>
      <td>5.391624e-03</td>
      <td>significant</td>
    </tr>
  </tbody>
</table>
</div>



**Conclusion** : For the features `operational_setting_3` and `sensor_measurement_14` there are significant differences in the survival function between the groups.

## Weibull Distribution

Lets fit the Weibull distribution as a parametric model for our time-to-event data. We use survival regression to make inferences about how covariates affect the survival function and select the best survival model for our data.


```python
# plt.xlabel callable bugfix
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)

# Import WeibullFitter class
from lifelines import WeibullFitter

# Instantiate WeibullFitter
wb = WeibullFitter()

# Fit data to wb
wb.fit(data_y["cycle"], data_y["status"])

# Plot survival function
wb.survival_function_.plot()
plt.title(f"time-to-failure with Weibull model")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel('cycles')

plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_65_0.png)
    



```python
threshold = 8100

# Mask for operational_setting_3
sm14 = (cleaned_data['sensor_measurement_14'] > threshold)

# Fit to sm14 greater than 8140
wb.fit(#durations=prison[parole]['week'], event_observed=prison[parole]['arrest']
      durations=cleaned_data[sm14]['cycle'], event_observed=cleaned_data[sm14]['status']
       #, label='exp'
      )

# Print rho
print("The rho_ parameter of group sm14 greater than 8140: ", wb.rho_)

# Fit to engines sm14 lower equal than 8140
wb.fit(durations=cleaned_data[~sm14]['cycle'], event_observed=cleaned_data[~sm14]['status'])

# Print rho
print("The rho_ parameter of group sm14 lower equal than 8140 survival function is: ", wb.rho_)
```

    The rho_ parameter of group sm14 greater than 8140:  3.278560362707543
    The rho_ parameter of group sm14 lower equal than 8140 survival function is:  3.8024815134050174
    

Both rho values are much larger than 1, which indicates that both groups' rate of event increases as time goes on.

## Multiple covariates with the Weibull Accelerated Failure Time model (Weibull AFT)


```python
# Import WeibullAFTFitter and instantiate
from lifelines import WeibullAFTFitter
aft = WeibullAFTFitter()

# Fit heart_patients data into aft
aft.fit(cleaned_data, duration_col='cycle', event_col='status')

# Print the summary
print(aft.summary)
```

                                        coef     exp(coef)   se(coef)  \
    param   covariate                                                   
    lambda_ operational_setting_3  -0.057131  9.444701e-01   0.014956   
            sensor_measurement_04  -0.009645  9.904010e-01   0.002959   
            sensor_measurement_11   0.475113  1.608197e+00   0.147451   
            sensor_measurement_14   0.003070  1.003074e+00   0.000585   
            sensor_measurement_16  -0.564594  5.685911e-01  13.080660   
            Intercept             -22.867489  1.171590e-10   4.716158   
    rho_    Intercept               1.306051  3.691566e+00   0.058606   
    
                                   coef lower 95%  coef upper 95%  \
    param   covariate                                               
    lambda_ operational_setting_3       -0.086444       -0.027818   
            sensor_measurement_04       -0.015446       -0.003845   
            sensor_measurement_11        0.186115        0.764112   
            sensor_measurement_14        0.001924        0.004215   
            sensor_measurement_16      -26.202217       25.073029   
            Intercept                  -32.110988      -13.623990   
    rho_    Intercept                    1.191185        1.420916   
    
                                   exp(coef) lower 95%  exp(coef) upper 95%  \
    param   covariate                                                         
    lambda_ operational_setting_3         9.171866e-01         9.725653e-01   
            sensor_measurement_04         9.846729e-01         9.961623e-01   
            sensor_measurement_11         1.204560e+00         2.147087e+00   
            sensor_measurement_14         1.001926e+00         1.004224e+00   
            sensor_measurement_16         4.173705e-12         7.746014e+10   
            Intercept                     1.133379e-14         1.211090e-06   
    rho_    Intercept                     3.290980e+00         4.140913e+00   
    
                                   cmp to          z              p    -log2(p)  
    param   covariate                                                            
    lambda_ operational_setting_3     0.0  -3.819968   1.334689e-04   12.871209  
            sensor_measurement_04     0.0  -3.259247   1.117084e-03    9.806046  
            sensor_measurement_11     0.0   3.222178   1.272201e-03    9.618458  
            sensor_measurement_14     0.0   5.250649   1.515641e-07   22.653569  
            sensor_measurement_16     0.0  -0.043162   9.655720e-01    0.050544  
            Intercept                 0.0  -4.848754   1.242393e-06   19.618447  
    rho_    Intercept                 0.0  22.285300  5.131147e-110  363.052809  
    


```python
# Calculate the exponential of sensor measurement 11 coefficient
exp_sm11 = np.exp(aft.params_.loc['lambda_'].loc['sensor_measurement_11'])
print('When sensor_measurement_11 increases by 1, the average survival duration changes by a factor of', exp_sm11)
```

    When sensor_measurement_11 increases by 1, the average survival duration changes by a factor of 1.608196633553678
    


```python
# Fit data to aft
aft.fit(cleaned_data, duration_col='cycle', event_col='status')

# Plot partial effects of prio
#aft.plot_partial_effects_on_outcome(  covariates='sensor_measurement_14',   values=np.arange(8090, 8170, 15))
aft.plot_partial_effects_on_outcome(  covariates='sensor_measurement_11',   values=np.arange(42, 45, 0.5))

plt.axvline(x=250,ls="--",c=".5")

plt.title(f"Weibull baseline model vs. varied values of sensor measurements 11")
plt.ylabel("survival probability")
plt.xlabel('cycles')
plt.show()

```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_73_0.png)
    


Visualized in this 'partial effects plot' above, we can see how sensor measurements 11 determine survival times significantly while every other covariate holding equal, even half a point difference of 42.5 vs. 43.0 at cycle 250 halves the survival probability.


```python
# Predict median of new data
aft_pred = aft.predict_median(cleaned_data)

# Print average predicted time to arrest
print("On average, the median number of cycles for an engine to fail is: ", np.mean(aft_pred))
```

    On average, the median number of cycles for an engine to fail is:  226.12247644680892
    

In case we want use predictions of our models we have to validate/invalidate our models, which can happen numerically and graphically.
The regular way to compare and validate different models is the Akaike information criterion (AIC) with which the best fitting model can be easily determined.


```python
from lifelines.fitters.exponential_fitter import ExponentialFitter 
from lifelines.fitters.log_normal_fitter import LogNormalFitter
from lifelines.fitters.log_logistic_fitter import LogLogisticFitter

# Instantiate each fitter
wb = WeibullFitter()
exp = ExponentialFitter()
lognorm = LogNormalFitter()
loglog = LogLogisticFitter()

# Fit to data
for model in [wb, exp, lognorm, loglog]:
      model.fit(durations=cleaned_data[sm14]['cycle'], event_observed=cleaned_data[sm14]['status'] )
      # Print AIC
      print(model.__class__.__name__, model.AIC_)
```

    WeibullFitter 1311.250877441705
    ExponentialFitter 1440.767027487693
    LogNormalFitter 1269.5021223158335
    LogLogisticFitter 1274.537494437834
    

#### How good is the fit visually?



```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.reshape(4,)

for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
    model.fit(durations=cleaned_data[sm14]['cycle'], event_observed=cleaned_data[sm14]['status'] )
    qq_plot(model, ax=axes[i])

plt.suptitle(f"QQ Plot of Fitted models of different distributions of those engines with Sensor Measurement 14 above threshold {threshold}" )
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_79_0.png)
    


The LogNormal Fitted Model fits the data best and reasonable well; minor tail separation from the diagonal line is expected and within acceptable bounds.


```python
data_x = data_x[data_x.columns[~data_x.columns.isin(["sensor_measurement_16=0.03"])]]
```

## Cox Propotional Hazard


The hazard function $ h(t) $ describes the probability that the event of interest will occur at a given time if individuals survive to that time. The hazard rates along this function are the instantaneous rates for the occurrence of the event.
$$
h(t) = h_{0}(t) e^{\beta_{1}x_{1} + \beta_{2}x_{2} + ... + \beta_{n}x_{n}}
$$

The Cox Propotional Hazard (CoxPH) model is a semiparametric model that will predict the value of the hazard rate which consists of the following components:

- Non-parametric $h_{0}(t)$ (*baseline hazard*) : Characterization of how the hazard function changes over time. This baseline hazard can also be interpreted as an intercept in the regression problem
-
Parametric $e^{\beta_{1}x_{1} + \beta_{2}x_{2} + ... + \beta_{n}x_{n}}$ : Characterization of how the hazard function changes based on covariate conditions (variables involved in the model)

The survival function based on the above model can be obtained through the hazard function, which are closely related:
$$
S(t) = e^{-\int{h(t)}} dt
$$

regularization alpha = 1e-4


```python
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Define model
estimator = CoxPHSurvivalAnalysis(verbose = 10)

# Fitting model
estimator.fit(data_x, data_y)
```

    iter      1: update = [-0.02672891  1.30951039  0.00897907 -6.14944208]
    iter      1: loss = 3.4407602869
    iter      2: update = [ 0.00043914 -0.0348166   0.00100936 -0.27206197]
    iter      2: loss = 3.4389450619
    iter      3: update = [-1.01083581e-05  6.09861343e-04 -3.68327372e-06 -1.80823588e-02]
    iter      3: loss = 3.4389378698
    iter      4: update = [ 1.34113910e-08 -7.92317762e-07  2.80717143e-09 -1.08672544e-04]
    iter      4: loss = 3.4389378694
    iter      4: optimization converged
    




    CoxPHSurvivalAnalysis(verbose=10)




```python
cleaned_data = cleaned_data.drop('sensor_measurement_16', axis=1)
```


```python
# Import CoxPHFitter class
from lifelines import CoxPHFitter

# Instantiate CoxPHFitter class cph
cph = CoxPHFitter(
    #penalizer=0.001
)

# Fit cph to data
cph.fit(df=cleaned_data, duration_col="cycle", event_col="status")

# Print model summary
print(cph.summary)
```

                               coef  exp(coef)  se(coef)  coef lower 95%  \
    covariate                                                              
    sensor_measurement_04  0.026635   1.026993  0.009872        0.007286   
    sensor_measurement_11 -1.291693   0.274805  0.515289       -2.301640   
    sensor_measurement_14 -0.010052   0.989998  0.002043       -0.014057   
    operational_setting_3  0.162776   1.176773  0.053030        0.058840   
    
                           coef upper 95%  exp(coef) lower 95%  \
    covariate                                                    
    sensor_measurement_04        0.045984             1.007312   
    sensor_measurement_11       -0.281746             0.100095   
    sensor_measurement_14       -0.006047             0.986041   
    operational_setting_3        0.266712             1.060605   
    
                           exp(coef) upper 95%  cmp to         z             p  \
    covariate                                                                    
    sensor_measurement_04             1.047058     0.0  2.697954  6.976709e-03   
    sensor_measurement_11             0.754465     0.0 -2.506737  1.218513e-02   
    sensor_measurement_14             0.993971     0.0 -4.919302  8.685343e-07   
    operational_setting_3             1.305664     0.0  3.069527  2.143978e-03   
    
                            -log2(p)  
    covariate                         
    sensor_measurement_04   7.163238  
    sensor_measurement_11   6.358734  
    sensor_measurement_14  20.134914  
    operational_setting_3   8.865494  
    

The hazard ratio is `exp(coef)` which is e to the power of a coefficient. It indicates how much the baseline hazard changes with a one-unit change in the covariate.

So with one unit increase in `operational_setting_3` the hazard changes by a factor of 1.137, a 13.7% increase compared to the baseline hazard.

All four covariates are statistically significant as per column p, indicating that there is a strong correlation between the changes in all covariates and the hazards.


```python
# Coef model
pd.Series(estimator.coef_, index = data_x.columns)
```




    sensor_measurement_04          0.026300
    sensor_measurement_11         -1.275303
    sensor_measurement_14         -0.009985
    operational_setting_3=100.0    6.439695
    dtype: float64



**Interpretation:**

- Every one unit increase in `sensor_measurement_04` will increase the hazard rate by $e^{0.02} = 1.02$ assuming no changes in other covariates
- Every one unit increase in `sensor_measurement_11` will reduce the hazard rate by $e^{-1.13} = 0.
32$ assuming no change in the other covariates
- Using `operational_setting_3` = 100 will increase the hazard rate by $e^{6.26} = 523.21$ compared to `operational_setting_3` = 60




```python
# Assign summary to summary_df
summary_df = cph.summary

# Create new column of survival time ratios
summary_df["surv_ratio"] = 1 / summary_df['exp(coef)']

summary_df
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
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>cmp to</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
      <th>surv_ratio</th>
    </tr>
    <tr>
      <th>covariate</th>
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
      <th>sensor_measurement_04</th>
      <td>0.026635</td>
      <td>1.026993</td>
      <td>0.009872</td>
      <td>0.007286</td>
      <td>0.045984</td>
      <td>1.007312</td>
      <td>1.047058</td>
      <td>0.0</td>
      <td>2.697954</td>
      <td>6.976709e-03</td>
      <td>7.163238</td>
      <td>0.973717</td>
    </tr>
    <tr>
      <th>sensor_measurement_11</th>
      <td>-1.291693</td>
      <td>0.274805</td>
      <td>0.515289</td>
      <td>-2.301640</td>
      <td>-0.281746</td>
      <td>0.100095</td>
      <td>0.754465</td>
      <td>0.0</td>
      <td>-2.506737</td>
      <td>1.218513e-02</td>
      <td>6.358734</td>
      <td>3.638942</td>
    </tr>
    <tr>
      <th>sensor_measurement_14</th>
      <td>-0.010052</td>
      <td>0.989998</td>
      <td>0.002043</td>
      <td>-0.014057</td>
      <td>-0.006047</td>
      <td>0.986041</td>
      <td>0.993971</td>
      <td>0.0</td>
      <td>-4.919302</td>
      <td>8.685343e-07</td>
      <td>20.134914</td>
      <td>1.010103</td>
    </tr>
    <tr>
      <th>operational_setting_3</th>
      <td>0.162776</td>
      <td>1.176773</td>
      <td>0.053030</td>
      <td>0.058840</td>
      <td>0.266712</td>
      <td>1.060605</td>
      <td>1.305664</td>
      <td>0.0</td>
      <td>3.069527</td>
      <td>2.143978e-03</td>
      <td>8.865494</td>
      <td>0.849782</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print surv_ratio for prio
survivalratio = summary_df.loc['sensor_measurement_04', "surv_ratio"]
survivalratio
```




    0.973716525421316




```python
TimeToFailureChange = survivalratio - 1
TimeToFailureChange
```




    -0.026283474578683963




```python
print(f'Time-to-failure changes by {TimeToFailureChange * 100:.2f} %')
```

    Time-to-failure changes by -2.63 %
    

So the impact on survival time with a one-unit increase in sensor_measurement_04 is more than negative 2 percent.


```python
covars = list(summary_df["surv_ratio"].index)
covars
```




    ['sensor_measurement_04',
     'sensor_measurement_11',
     'sensor_measurement_14',
     'operational_setting_3']




```python
sorted(covars)
```




    ['operational_setting_3',
     'sensor_measurement_04',
     'sensor_measurement_11',
     'sensor_measurement_14']




```python
# Plot partial effects
cph.plot_partial_effects_on_outcome(covariates=sorted(covars),
         values=[[100.0,1140,42,8080], 
                 [100.0,1130,43,8100], 
                 [100.0,1150,45,8120],
                 [60,   1150,40,8080]])

# Show plot
plt.title("Indiviual survival curves of four prototype Turbofan Jet engines with specific covariate properties - Time to Failure Estimation" )
plt.ylabel("est. probability of survival")
#plt.xlabel("cycles")
plt.legend(loc="upper right")
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_99_0.png)
    



```python
# Check PH assumption
print(cph.check_assumptions(training_df=cleaned_data, p_value_threshold=0.05, show_plots=True))
```

    The ``p_value_threshold`` is set at 0.05. Even under the null hypothesis of no violations, some
    covariates will be below the threshold by chance. This is compounded when there are many covariates.
    Similarly, when there are lots of observations, even minor deviances from the proportional hazard
    assumption will be flagged.
    
    With that in mind, it's best to use a combination of statistical tests and visual tests to determine
    the most serious violations. Produce visual plots using ``check_assumptions(..., show_plots=True)``
    and looking for non-constant lines. See link [A] below for a full example.
    
    


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
  <tbody>
    <tr>
      <th>null_distribution</th>
      <td>chi squared</td>
    </tr>
    <tr>
      <th>degrees_of_freedom</th>
      <td>1</td>
    </tr>
    <tr>
      <th>model</th>
      <td>&lt;lifelines.CoxPHFitter: fitted with 260 total ...</td>
    </tr>
    <tr>
      <th>test_name</th>
      <td>proportional_hazard_test</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>test_statistic</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">operational_setting_3</th>
      <th>km</th>
      <td>0.87</td>
      <td>0.35</td>
      <td>1.51</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.95</td>
      <td>0.33</td>
      <td>1.60</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sensor_measurement_04</th>
      <th>km</th>
      <td>1.58</td>
      <td>0.21</td>
      <td>2.26</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>1.66</td>
      <td>0.20</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sensor_measurement_11</th>
      <th>km</th>
      <td>2.10</td>
      <td>0.15</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>2.20</td>
      <td>0.14</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sensor_measurement_14</th>
      <th>km</th>
      <td>8.35</td>
      <td>&lt;0.005</td>
      <td>8.02</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>8.12</td>
      <td>&lt;0.005</td>
      <td>7.84</td>
    </tr>
  </tbody>
</table>


    
    
    1. Variable 'sensor_measurement_14' failed the non-proportional test: p-value is 0.0039.
    
       Advice 1: the functional form of the variable 'sensor_measurement_14' might be incorrect. That
    is, there may be non-linear terms missing. The proportional hazard test used is very sensitive to
    incorrect functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'sensor_measurement_14' using pd.cut, and then specify it in
    `strata=['sensor_measurement_14', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
       Bootstrapping lowess lines. May take a moment...
    
    
    ---
    [A]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
    [B]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
    [C]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
    [D]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
    [E]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
    
    [[<AxesSubplot:xlabel='rank-transformed time\n(p=0.0044)'>, <AxesSubplot:xlabel='km-transformed time\n(p=0.0039)'>]]
    


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_100_3.png)
    


### Stratify sm14 and redo straified COXPH fitting


```python
cleaned_strata_data = cleaned_data.copy()
cleaned_data.sensor_measurement_14.describe()
```




    count     260.000000
    mean     8105.652538
    std        74.714146
    min      7850.960000
    25%      8081.580000
    50%      8116.755000
    75%      8148.730000
    max      8252.610000
    Name: sensor_measurement_14, dtype: float64




```python
cleaned_strata_data
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
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_11</th>
      <th>sensor_measurement_14</th>
      <th>operational_setting_3</th>
      <th>status</th>
      <th>cycle</th>
    </tr>
    <tr>
      <th>machine_name</th>
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
      <th>1</th>
      <td>1149.81</td>
      <td>42.77</td>
      <td>8066.19</td>
      <td>100.0</td>
      <td>True</td>
      <td>149</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1146.65</td>
      <td>42.64</td>
      <td>8110.26</td>
      <td>100.0</td>
      <td>False</td>
      <td>269</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1144.89</td>
      <td>42.85</td>
      <td>8082.25</td>
      <td>100.0</td>
      <td>True</td>
      <td>206</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1428.21</td>
      <td>48.25</td>
      <td>8215.14</td>
      <td>100.0</td>
      <td>False</td>
      <td>235</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1146.87</td>
      <td>42.67</td>
      <td>8151.36</td>
      <td>100.0</td>
      <td>True</td>
      <td>154</td>
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
      <th>256</th>
      <td>1425.84</td>
      <td>48.36</td>
      <td>8128.22</td>
      <td>100.0</td>
      <td>True</td>
      <td>163</td>
    </tr>
    <tr>
      <th>257</th>
      <td>1146.53</td>
      <td>42.70</td>
      <td>8171.37</td>
      <td>100.0</td>
      <td>False</td>
      <td>309</td>
    </tr>
    <tr>
      <th>258</th>
      <td>1326.79</td>
      <td>46.08</td>
      <td>8124.16</td>
      <td>100.0</td>
      <td>True</td>
      <td>143</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1149.97</td>
      <td>42.73</td>
      <td>8075.91</td>
      <td>100.0</td>
      <td>True</td>
      <td>205</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1145.52</td>
      <td>42.50</td>
      <td>8185.35</td>
      <td>100.0</td>
      <td>False</td>
      <td>316</td>
    </tr>
  </tbody>
</table>
<p>260 rows × 6 columns</p>
</div>




```python
cph2 = CoxPHFitter()
cph2.fit(df=cleaned_strata_data, duration_col="cycle", event_col="status", 
         #strata=['sm14_strata'], 
         formula="bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300) + sensor_measurement_04 + sensor_measurement_11 + operational_setting_3"
        )
```




    <lifelines.CoxPHFitter: fitted with 260 total observations, 81 right-censored observations>




```python
cph2.print_summary(3)
sns.set(rc = {'figure.figsize':(7,7)})
cph2.plot()
plt.show()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'cycle'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'status'</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>260</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>179</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-891.762</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2022-09-17 08:02:35 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300)[1]</th>
      <td>0.931</td>
      <td>2.537</td>
      <td>2.994</td>
      <td>-4.937</td>
      <td>6.799</td>
      <td>0.007</td>
      <td>896.764</td>
      <td>0.000</td>
      <td>0.311</td>
      <td>0.756</td>
      <td>0.404</td>
    </tr>
    <tr>
      <th>bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300)[2]</th>
      <td>-4.767</td>
      <td>0.009</td>
      <td>2.662</td>
      <td>-9.984</td>
      <td>0.451</td>
      <td>0.000</td>
      <td>1.570</td>
      <td>0.000</td>
      <td>-1.791</td>
      <td>0.073</td>
      <td>3.769</td>
    </tr>
    <tr>
      <th>bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300)[3]</th>
      <td>-2.244</td>
      <td>0.106</td>
      <td>2.345</td>
      <td>-6.840</td>
      <td>2.351</td>
      <td>0.001</td>
      <td>10.500</td>
      <td>0.000</td>
      <td>-0.957</td>
      <td>0.338</td>
      <td>1.563</td>
    </tr>
    <tr>
      <th>operational_setting_3[T.100.0]</th>
      <td>6.298</td>
      <td>543.528</td>
      <td>2.649</td>
      <td>1.107</td>
      <td>11.489</td>
      <td>3.025</td>
      <td>97671.295</td>
      <td>0.000</td>
      <td>2.378</td>
      <td>0.017</td>
      <td>5.844</td>
    </tr>
    <tr>
      <th>sensor_measurement_04</th>
      <td>0.027</td>
      <td>1.027</td>
      <td>0.010</td>
      <td>0.008</td>
      <td>0.046</td>
      <td>1.008</td>
      <td>1.047</td>
      <td>0.000</td>
      <td>2.725</td>
      <td>0.006</td>
      <td>7.281</td>
    </tr>
    <tr>
      <th>sensor_measurement_11</th>
      <td>-1.298</td>
      <td>0.273</td>
      <td>0.514</td>
      <td>-2.306</td>
      <td>-0.290</td>
      <td>0.100</td>
      <td>0.749</td>
      <td>0.000</td>
      <td>-2.523</td>
      <td>0.012</td>
      <td>6.425</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.650</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>1795.525</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>39.294 on 6 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>20.605</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_107_1.png)
    



```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cph2.check_assumptions(cleaned_strata_data)
```

    The ``p_value_threshold`` is set at 0.01. Even under the null hypothesis of no violations, some
    covariates will be below the threshold by chance. This is compounded when there are many covariates.
    Similarly, when there are lots of observations, even minor deviances from the proportional hazard
    assumption will be flagged.
    
    With that in mind, it's best to use a combination of statistical tests and visual tests to determine
    the most serious violations. Produce visual plots using ``check_assumptions(..., show_plots=True)``
    and looking for non-constant lines. See link [A] below for a full example.
    
    


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
  <tbody>
    <tr>
      <th>null_distribution</th>
      <td>chi squared</td>
    </tr>
    <tr>
      <th>degrees_of_freedom</th>
      <td>1</td>
    </tr>
    <tr>
      <th>model</th>
      <td>&lt;lifelines.CoxPHFitter: fitted with 260 total ...</td>
    </tr>
    <tr>
      <th>test_name</th>
      <td>proportional_hazard_test</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>test_statistic</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300)[1]</th>
      <th>km</th>
      <td>3.38</td>
      <td>0.07</td>
      <td>3.92</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>3.38</td>
      <td>0.07</td>
      <td>3.92</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300)[2]</th>
      <th>km</th>
      <td>0.31</td>
      <td>0.58</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.29</td>
      <td>0.59</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300)[3]</th>
      <th>km</th>
      <td>6.63</td>
      <td>0.01</td>
      <td>6.64</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>6.59</td>
      <td>0.01</td>
      <td>6.61</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">operational_setting_3[T.100.0]</th>
      <th>km</th>
      <td>0.06</td>
      <td>0.80</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.08</td>
      <td>0.77</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sensor_measurement_04</th>
      <th>km</th>
      <td>1.70</td>
      <td>0.19</td>
      <td>2.38</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>1.78</td>
      <td>0.18</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sensor_measurement_11</th>
      <th>km</th>
      <td>2.21</td>
      <td>0.14</td>
      <td>2.87</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>2.31</td>
      <td>0.13</td>
      <td>2.96</td>
    </tr>
  </tbody>
</table>


    
    
    1. Variable 'bs(sensor_measurement_14, df=3, lower_bound=7850, upper_bound=8300)[3]' failed the non-proportional test: p-value is 0.0100.
    
       Advice 1: the functional form of the variable 'bs(sensor_measurement_14, df=3, lower_bound=7850,
    upper_bound=8300)[3]' might be incorrect. That is, there may be non-linear terms missing. The
    proportional hazard test used is very sensitive to incorrect functional forms. See documentation in
    link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'bs(sensor_measurement_14, df=3, lower_bound=7850,
    upper_bound=8300)[3]' using pd.cut, and then specify it in `strata=['bs(sensor_measurement_14, df=3,
    lower_bound=7850, upper_bound=8300)[3]', ...]` in the call in `.fit`. See documentation in link [B]
    below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    ---
    [A]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
    [B]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
    [C]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
    [D]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
    [E]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
    
    




    []



#### Time to Failure prediction for a new set of turbofan jets engines


```python
# Select engines that have not failed yet
engines_still_working = cleaned_strata_data.loc[cleaned_strata_data['status'] == False].copy()
```


```python
engines_still_working.loc[:, 'age_cycles'] = 200
engines_still_working
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
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_11</th>
      <th>sensor_measurement_14</th>
      <th>operational_setting_3</th>
      <th>status</th>
      <th>cycle</th>
      <th>age_cycles</th>
    </tr>
    <tr>
      <th>machine_name</th>
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
      <th>2</th>
      <td>1146.65</td>
      <td>42.64</td>
      <td>8110.26</td>
      <td>100.0</td>
      <td>False</td>
      <td>269</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1428.21</td>
      <td>48.25</td>
      <td>8215.14</td>
      <td>100.0</td>
      <td>False</td>
      <td>235</td>
      <td>200</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1144.57</td>
      <td>42.83</td>
      <td>8167.76</td>
      <td>100.0</td>
      <td>False</td>
      <td>271</td>
      <td>200</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1326.77</td>
      <td>46.25</td>
      <td>8132.04</td>
      <td>100.0</td>
      <td>False</td>
      <td>249</td>
      <td>200</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1146.18</td>
      <td>42.67</td>
      <td>8055.61</td>
      <td>100.0</td>
      <td>False</td>
      <td>227</td>
      <td>200</td>
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
      <th>251</th>
      <td>1145.99</td>
      <td>42.61</td>
      <td>8144.66</td>
      <td>100.0</td>
      <td>False</td>
      <td>266</td>
      <td>200</td>
    </tr>
    <tr>
      <th>254</th>
      <td>1075.12</td>
      <td>37.46</td>
      <td>7855.14</td>
      <td>60.0</td>
      <td>False</td>
      <td>260</td>
      <td>200</td>
    </tr>
    <tr>
      <th>255</th>
      <td>1143.87</td>
      <td>42.79</td>
      <td>8091.37</td>
      <td>100.0</td>
      <td>False</td>
      <td>340</td>
      <td>200</td>
    </tr>
    <tr>
      <th>257</th>
      <td>1146.53</td>
      <td>42.70</td>
      <td>8171.37</td>
      <td>100.0</td>
      <td>False</td>
      <td>309</td>
      <td>200</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1145.52</td>
      <td>42.50</td>
      <td>8185.35</td>
      <td>100.0</td>
      <td>False</td>
      <td>316</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 7 columns</p>
</div>




```python
# Existing durations of employees that have not churned
engines_still_working_last_obs = engines_still_working['age_cycles']
```


```python
# Predict survival function conditional on existing durations
cph2.predict_survival_function(engines_still_working, 
                              conditional_after=engines_still_working_last_obs)
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
      <th>2</th>
      <th>4</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>31</th>
      <th>32</th>
      <th>41</th>
      <th>42</th>
      <th>47</th>
      <th>...</th>
      <th>240</th>
      <th>242</th>
      <th>243</th>
      <th>245</th>
      <th>248</th>
      <th>251</th>
      <th>254</th>
      <th>255</th>
      <th>257</th>
      <th>260</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>129.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>133.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>135.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>136.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>343.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>344.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>347.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>365.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
    <tr>
      <th>378.0</th>
      <td>0.638475</td>
      <td>0.749801</td>
      <td>0.82612</td>
      <td>0.662427</td>
      <td>0.422205</td>
      <td>0.704433</td>
      <td>0.761802</td>
      <td>0.532168</td>
      <td>0.660271</td>
      <td>0.815766</td>
      <td>...</td>
      <td>0.600644</td>
      <td>0.770246</td>
      <td>0.842319</td>
      <td>0.756958</td>
      <td>0.688897</td>
      <td>0.727469</td>
      <td>0.406703</td>
      <td>0.649048</td>
      <td>0.79238</td>
      <td>0.762147</td>
    </tr>
  </tbody>
</table>
<p>133 rows × 81 columns</p>
</div>




```python
# Predict median remaining times for current employees
pred = cph2.predict_median(engines_still_working, conditional_after=engines_still_working_last_obs)

# Print the smallest median remaining time
print(pred)
```

    2        inf
    4        inf
    11       inf
    12       inf
    13     128.0
           ...  
    251      inf
    254    128.0
    255      inf
    257      inf
    260      inf
    Name: 0.5, Length: 81, dtype: float64
    


```python
numbers =pred != np.inf 
numbers.sum()
```




    10



So far we have ignored the time series component in this data set reviewed the problem as a classical surivival problem where usually covariates do not vary or are not recorded.

Now we 

## Predict Remaining Useful Lifetime (RUL) with a CoxTimeVary Model

The fitter of the CoxTimeVarying model, which can take advantage of the time series nature of the data, as it is able to consider multiple observations for each engine.

The downside of this model is that its results are less intuitive to interpret. In general, higher partial hazards indicate a greater risk of breakdown, but this **is not** a direct indicator of remaining useful lifetime (time-to-failure).



```python
#train_cols = index_names + remaining_sensors + ['start', 'breakdown']
#predict_cols = ['time_cycles'] + remaining_sensors + ['start', 'breakdown']  # breakdown value will be 0
```


```python
timevary_df = timeseries_df.copy()
timevary_df_test = timeseries_df_test.copy()
timevary_df.head()
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>...</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>34.9983</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>555.32</td>
      <td>1358.61</td>
      <td>1137.23</td>
      <td>5.48</td>
      <td>...</td>
      <td>183.06</td>
      <td>2387.72</td>
      <td>8048.56</td>
      <td>9.3461</td>
      <td>0.02</td>
      <td>334</td>
      <td>2223</td>
      <td>100.00</td>
      <td>14.73</td>
      <td>8.8071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>41.9982</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.90</td>
      <td>1353.22</td>
      <td>1125.78</td>
      <td>3.91</td>
      <td>...</td>
      <td>130.42</td>
      <td>2387.66</td>
      <td>8072.30</td>
      <td>9.3774</td>
      <td>0.02</td>
      <td>330</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.41</td>
      <td>6.2665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>24.9988</td>
      <td>0.6218</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.31</td>
      <td>1256.76</td>
      <td>1047.45</td>
      <td>7.05</td>
      <td>...</td>
      <td>164.22</td>
      <td>2028.03</td>
      <td>7864.87</td>
      <td>10.8941</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.08</td>
      <td>8.6723</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>42.0077</td>
      <td>0.8416</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.51</td>
      <td>1354.03</td>
      <td>1126.38</td>
      <td>3.91</td>
      <td>...</td>
      <td>130.72</td>
      <td>2387.61</td>
      <td>8068.66</td>
      <td>9.3528</td>
      <td>0.02</td>
      <td>329</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.59</td>
      <td>6.4701</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>25.0005</td>
      <td>0.6203</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.07</td>
      <td>1257.71</td>
      <td>1047.93</td>
      <td>7.05</td>
      <td>...</td>
      <td>164.31</td>
      <td>2028.00</td>
      <td>7861.23</td>
      <td>10.8963</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.13</td>
      <td>8.5286</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
timevary_df.groupby(by="machine_name")['cycle'].max()
```




    machine_name
    1      149
    2      269
    3      206
    4      235
    5      154
          ... 
    256    163
    257    309
    258    143
    259    205
    260    316
    Name: cycle, Length: 260, dtype: int64




```python
# RUL
RUL_timevary_df = timevary_df.merge(timevary_df.groupby(by="machine_name")['cycle'].max().to_frame(name='fail_cycle'), left_on='machine_name', right_index=True)
RUL_timevary_df_test = timevary_df_test.merge(timevary_df_test.groupby(by="machine_name")['cycle'].max().to_frame(name='fail_cycle'), left_on='machine_name', right_index=True)
RUL_timevary_df 
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>...</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
      <th>fail_cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>34.9983</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>555.32</td>
      <td>1358.61</td>
      <td>1137.23</td>
      <td>5.48</td>
      <td>...</td>
      <td>2387.72</td>
      <td>8048.56</td>
      <td>9.3461</td>
      <td>0.02</td>
      <td>334</td>
      <td>2223</td>
      <td>100.00</td>
      <td>14.73</td>
      <td>8.8071</td>
      <td>149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>41.9982</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.90</td>
      <td>1353.22</td>
      <td>1125.78</td>
      <td>3.91</td>
      <td>...</td>
      <td>2387.66</td>
      <td>8072.30</td>
      <td>9.3774</td>
      <td>0.02</td>
      <td>330</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.41</td>
      <td>6.2665</td>
      <td>149</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>24.9988</td>
      <td>0.6218</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.31</td>
      <td>1256.76</td>
      <td>1047.45</td>
      <td>7.05</td>
      <td>...</td>
      <td>2028.03</td>
      <td>7864.87</td>
      <td>10.8941</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.08</td>
      <td>8.6723</td>
      <td>149</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>42.0077</td>
      <td>0.8416</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.51</td>
      <td>1354.03</td>
      <td>1126.38</td>
      <td>3.91</td>
      <td>...</td>
      <td>2387.61</td>
      <td>8068.66</td>
      <td>9.3528</td>
      <td>0.02</td>
      <td>329</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.59</td>
      <td>6.4701</td>
      <td>149</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>25.0005</td>
      <td>0.6203</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.07</td>
      <td>1257.71</td>
      <td>1047.93</td>
      <td>7.05</td>
      <td>...</td>
      <td>2028.00</td>
      <td>7861.23</td>
      <td>10.8963</td>
      <td>0.02</td>
      <td>309</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.13</td>
      <td>8.5286</td>
      <td>149</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>53754</th>
      <td>260</td>
      <td>312</td>
      <td>20.0037</td>
      <td>0.7000</td>
      <td>100.0</td>
      <td>491.19</td>
      <td>608.79</td>
      <td>1495.60</td>
      <td>1269.51</td>
      <td>9.35</td>
      <td>...</td>
      <td>2389.02</td>
      <td>8169.64</td>
      <td>9.3035</td>
      <td>0.03</td>
      <td>369</td>
      <td>2324</td>
      <td>100.00</td>
      <td>24.36</td>
      <td>14.5189</td>
      <td>316</td>
    </tr>
    <tr>
      <th>53755</th>
      <td>260</td>
      <td>313</td>
      <td>10.0022</td>
      <td>0.2510</td>
      <td>100.0</td>
      <td>489.05</td>
      <td>605.81</td>
      <td>1514.32</td>
      <td>1324.12</td>
      <td>10.52</td>
      <td>...</td>
      <td>2388.42</td>
      <td>8245.36</td>
      <td>8.7586</td>
      <td>0.03</td>
      <td>374</td>
      <td>2319</td>
      <td>100.00</td>
      <td>28.10</td>
      <td>16.9454</td>
      <td>316</td>
    </tr>
    <tr>
      <th>53756</th>
      <td>260</td>
      <td>314</td>
      <td>25.0041</td>
      <td>0.6200</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.48</td>
      <td>1276.24</td>
      <td>1057.92</td>
      <td>7.05</td>
      <td>...</td>
      <td>2030.33</td>
      <td>7971.25</td>
      <td>11.0657</td>
      <td>0.02</td>
      <td>310</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.19</td>
      <td>8.5503</td>
      <td>316</td>
    </tr>
    <tr>
      <th>53757</th>
      <td>260</td>
      <td>315</td>
      <td>25.0033</td>
      <td>0.6220</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>537.84</td>
      <td>1272.95</td>
      <td>1066.30</td>
      <td>7.05</td>
      <td>...</td>
      <td>2030.35</td>
      <td>7972.47</td>
      <td>11.0537</td>
      <td>0.02</td>
      <td>311</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.05</td>
      <td>8.3729</td>
      <td>316</td>
    </tr>
    <tr>
      <th>53758</th>
      <td>260</td>
      <td>316</td>
      <td>35.0036</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>556.64</td>
      <td>1374.61</td>
      <td>1145.52</td>
      <td>5.48</td>
      <td>...</td>
      <td>2390.38</td>
      <td>8185.35</td>
      <td>9.3998</td>
      <td>0.02</td>
      <td>338</td>
      <td>2223</td>
      <td>100.00</td>
      <td>14.75</td>
      <td>8.8446</td>
      <td>316</td>
    </tr>
  </tbody>
</table>
<p>53759 rows × 27 columns</p>
</div>




```python
RUL_timevary_df['start'] = RUL_timevary_df['cycle'] - 1
RUL_timevary_df_test['start'] = RUL_timevary_df_test['cycle'] - 1

#timevary_df['status'] = timevary_df['cycle'].apply(lambda x: False if x > timevary_df['fail_cycle'] else True)
RUL_timevary_df['status'] = np.where(RUL_timevary_df['cycle']>=RUL_timevary_df ['fail_cycle'], True, False)
RUL_timevary_df_test['status'] = np.where(RUL_timevary_df_test['cycle']>=RUL_timevary_df_test['fail_cycle'], True, False)
```


```python
# Calculate remaining useful life for each row
RUL_timevary_df["RUL"] = RUL_timevary_df["fail_cycle"] - RUL_timevary_df["cycle"]
RUL_timevary_df_test["RUL"] = RUL_timevary_df_test["fail_cycle"] - RUL_timevary_df_test["cycle"]
RUL_timevary_df_test
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_01</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_05</th>
      <th>...</th>
      <th>sensor_measurement_16</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_18</th>
      <th>sensor_measurement_19</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
      <th>fail_cycle</th>
      <th>start</th>
      <th>status</th>
      <th>RUL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>9.9987</td>
      <td>0.2502</td>
      <td>100.0</td>
      <td>489.05</td>
      <td>605.03</td>
      <td>1497.17</td>
      <td>1304.99</td>
      <td>10.52</td>
      <td>...</td>
      <td>0.03</td>
      <td>369</td>
      <td>2319</td>
      <td>100.00</td>
      <td>28.42</td>
      <td>17.1551</td>
      <td>258</td>
      <td>0</td>
      <td>False</td>
      <td>257</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>20.0026</td>
      <td>0.7000</td>
      <td>100.0</td>
      <td>491.19</td>
      <td>607.82</td>
      <td>1481.20</td>
      <td>1246.11</td>
      <td>9.35</td>
      <td>...</td>
      <td>0.02</td>
      <td>364</td>
      <td>2324</td>
      <td>100.00</td>
      <td>24.29</td>
      <td>14.8039</td>
      <td>258</td>
      <td>1</td>
      <td>False</td>
      <td>256</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>35.0045</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>556.00</td>
      <td>1359.08</td>
      <td>1128.36</td>
      <td>5.48</td>
      <td>...</td>
      <td>0.02</td>
      <td>333</td>
      <td>2223</td>
      <td>100.00</td>
      <td>14.98</td>
      <td>8.9125</td>
      <td>258</td>
      <td>2</td>
      <td>False</td>
      <td>255</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>42.0066</td>
      <td>0.8410</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>550.17</td>
      <td>1349.69</td>
      <td>1127.89</td>
      <td>3.91</td>
      <td>...</td>
      <td>0.02</td>
      <td>332</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.35</td>
      <td>6.4181</td>
      <td>258</td>
      <td>3</td>
      <td>False</td>
      <td>254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>24.9985</td>
      <td>0.6213</td>
      <td>60.0</td>
      <td>462.54</td>
      <td>536.72</td>
      <td>1253.18</td>
      <td>1050.69</td>
      <td>7.05</td>
      <td>...</td>
      <td>0.02</td>
      <td>305</td>
      <td>1915</td>
      <td>84.93</td>
      <td>14.31</td>
      <td>8.5740</td>
      <td>258</td>
      <td>4</td>
      <td>False</td>
      <td>253</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>33986</th>
      <td>259</td>
      <td>119</td>
      <td>35.0015</td>
      <td>0.8403</td>
      <td>100.0</td>
      <td>449.44</td>
      <td>555.56</td>
      <td>1366.01</td>
      <td>1129.47</td>
      <td>5.48</td>
      <td>...</td>
      <td>0.02</td>
      <td>334</td>
      <td>2223</td>
      <td>100.00</td>
      <td>14.94</td>
      <td>8.9065</td>
      <td>123</td>
      <td>118</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33987</th>
      <td>259</td>
      <td>120</td>
      <td>42.0066</td>
      <td>0.8405</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.42</td>
      <td>1351.13</td>
      <td>1123.86</td>
      <td>3.91</td>
      <td>...</td>
      <td>0.02</td>
      <td>332</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.57</td>
      <td>6.4075</td>
      <td>123</td>
      <td>119</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>33988</th>
      <td>259</td>
      <td>121</td>
      <td>42.0061</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.65</td>
      <td>1349.14</td>
      <td>1118.91</td>
      <td>3.91</td>
      <td>...</td>
      <td>0.02</td>
      <td>331</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.57</td>
      <td>6.4805</td>
      <td>123</td>
      <td>120</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33989</th>
      <td>259</td>
      <td>122</td>
      <td>0.0024</td>
      <td>0.0003</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.58</td>
      <td>1589.61</td>
      <td>1408.16</td>
      <td>14.62</td>
      <td>...</td>
      <td>0.03</td>
      <td>393</td>
      <td>2388</td>
      <td>100.00</td>
      <td>39.08</td>
      <td>23.3589</td>
      <td>123</td>
      <td>121</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33990</th>
      <td>259</td>
      <td>123</td>
      <td>42.0033</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>445.00</td>
      <td>549.77</td>
      <td>1342.50</td>
      <td>1126.96</td>
      <td>3.91</td>
      <td>...</td>
      <td>0.02</td>
      <td>331</td>
      <td>2212</td>
      <td>100.00</td>
      <td>10.63</td>
      <td>6.3480</td>
      <td>123</td>
      <td>122</td>
      <td>True</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>33991 rows × 30 columns</p>
</div>




```python
from lifelines import CoxTimeVaryingFitter

#RUL_timevary_df_training = RUL_timevary_df.drop(['fail_cycle', 'RUL'], axis=1)
# drop additional 
RUL_timevary_df_training = RUL_timevary_df[['machine_name', 'cycle', 'operational_setting_1',
       'operational_setting_2', 'operational_setting_3',
        'sensor_measurement_02',
       'sensor_measurement_03', 'sensor_measurement_04',
      
       'sensor_measurement_07', 'sensor_measurement_08',
       'sensor_measurement_09', 'sensor_measurement_10',
       'sensor_measurement_11', 'sensor_measurement_12',
       'sensor_measurement_13', 'sensor_measurement_14',
       'sensor_measurement_15', 
       'sensor_measurement_17',  'sensor_measurement_20',
       'sensor_measurement_21', 'start', 'status']]

```


```python
ctv = CoxTimeVaryingFitter()
ctv.fit(RUL_timevary_df_training, id_col="machine_name", event_col='status', 
        start_col='start', stop_col='cycle', show_progress=True, step_size=1)
```

    Iteration 50: norm_delta = 2.70442, step_size = 0.23075, ll = -525.47872, newton_decrement = 0.31545, seconds_since_start = 3.4Convergence failed. See any warning messages.
    




    <lifelines.CoxTimeVaryingFitter: fitted with 53759 periods, 260 subjects, 260 events>




```python
ctv.print_summary()
plt.figure(figsize=(10,5))
ctv.plot()
plt.show()
plt.close()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxTimeVaryingFitter</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'status'</td>
    </tr>
    <tr>
      <th>number of subjects</th>
      <td>260</td>
    </tr>
    <tr>
      <th>number of periods</th>
      <td>53759</td>
    </tr>
    <tr>
      <th>number of events</th>
      <td>260</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-525.48</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2022-09-17 08:02:36 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>operational_setting_1</th>
      <td>-1.02</td>
      <td>0.36</td>
      <td>0.22</td>
      <td>-1.45</td>
      <td>-0.58</td>
      <td>0.23</td>
      <td>0.56</td>
      <td>0.00</td>
      <td>-4.58</td>
      <td>&lt;0.005</td>
      <td>17.71</td>
    </tr>
    <tr>
      <th>operational_setting_2</th>
      <td>30.86</td>
      <td>2.51e+13</td>
      <td>11.87</td>
      <td>7.59</td>
      <td>54.12</td>
      <td>1988.10</td>
      <td>3.18e+23</td>
      <td>0.00</td>
      <td>2.60</td>
      <td>0.01</td>
      <td>6.74</td>
    </tr>
    <tr>
      <th>operational_setting_3</th>
      <td>-14.26</td>
      <td>0.00</td>
      <td>2.21</td>
      <td>-18.59</td>
      <td>-9.92</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-6.44</td>
      <td>&lt;0.005</td>
      <td>32.97</td>
    </tr>
    <tr>
      <th>sensor_measurement_02</th>
      <td>1.50</td>
      <td>4.49</td>
      <td>0.25</td>
      <td>1.01</td>
      <td>1.99</td>
      <td>2.76</td>
      <td>7.31</td>
      <td>0.00</td>
      <td>6.04</td>
      <td>&lt;0.005</td>
      <td>29.31</td>
    </tr>
    <tr>
      <th>sensor_measurement_03</th>
      <td>0.09</td>
      <td>1.10</td>
      <td>0.02</td>
      <td>0.06</td>
      <td>0.13</td>
      <td>1.06</td>
      <td>1.14</td>
      <td>0.00</td>
      <td>5.30</td>
      <td>&lt;0.005</td>
      <td>23.00</td>
    </tr>
    <tr>
      <th>sensor_measurement_04</th>
      <td>0.11</td>
      <td>1.11</td>
      <td>0.02</td>
      <td>0.07</td>
      <td>0.14</td>
      <td>1.08</td>
      <td>1.15</td>
      <td>0.00</td>
      <td>6.38</td>
      <td>&lt;0.005</td>
      <td>32.36</td>
    </tr>
    <tr>
      <th>sensor_measurement_07</th>
      <td>0.11</td>
      <td>1.11</td>
      <td>0.14</td>
      <td>-0.17</td>
      <td>0.38</td>
      <td>0.85</td>
      <td>1.46</td>
      <td>0.00</td>
      <td>0.77</td>
      <td>0.44</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>sensor_measurement_08</th>
      <td>-1.42</td>
      <td>0.24</td>
      <td>0.14</td>
      <td>-1.69</td>
      <td>-1.15</td>
      <td>0.18</td>
      <td>0.32</td>
      <td>0.00</td>
      <td>-10.32</td>
      <td>&lt;0.005</td>
      <td>80.56</td>
    </tr>
    <tr>
      <th>sensor_measurement_09</th>
      <td>0.03</td>
      <td>1.03</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>1.06</td>
      <td>0.00</td>
      <td>2.10</td>
      <td>0.04</td>
      <td>4.80</td>
    </tr>
    <tr>
      <th>sensor_measurement_10</th>
      <td>-0.45</td>
      <td>0.64</td>
      <td>29.28</td>
      <td>-57.83</td>
      <td>56.94</td>
      <td>0.00</td>
      <td>5.34e+24</td>
      <td>0.00</td>
      <td>-0.02</td>
      <td>0.99</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>sensor_measurement_11</th>
      <td>7.20</td>
      <td>1344.92</td>
      <td>0.66</td>
      <td>5.91</td>
      <td>8.50</td>
      <td>367.50</td>
      <td>4921.91</td>
      <td>0.00</td>
      <td>10.88</td>
      <td>&lt;0.005</td>
      <td>89.22</td>
    </tr>
    <tr>
      <th>sensor_measurement_12</th>
      <td>0.06</td>
      <td>1.07</td>
      <td>0.15</td>
      <td>-0.23</td>
      <td>0.36</td>
      <td>0.79</td>
      <td>1.44</td>
      <td>0.00</td>
      <td>0.42</td>
      <td>0.67</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>sensor_measurement_13</th>
      <td>2.61</td>
      <td>13.62</td>
      <td>0.28</td>
      <td>2.07</td>
      <td>3.16</td>
      <td>7.91</td>
      <td>23.46</td>
      <td>0.00</td>
      <td>9.41</td>
      <td>&lt;0.005</td>
      <td>67.49</td>
    </tr>
    <tr>
      <th>sensor_measurement_14</th>
      <td>-0.04</td>
      <td>0.96</td>
      <td>0.02</td>
      <td>-0.07</td>
      <td>-0.01</td>
      <td>0.93</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>-2.65</td>
      <td>0.01</td>
      <td>6.97</td>
    </tr>
    <tr>
      <th>sensor_measurement_15</th>
      <td>18.04</td>
      <td>6.82e+07</td>
      <td>3.31</td>
      <td>11.55</td>
      <td>24.52</td>
      <td>1.04e+05</td>
      <td>4.46e+10</td>
      <td>0.00</td>
      <td>5.45</td>
      <td>&lt;0.005</td>
      <td>24.27</td>
    </tr>
    <tr>
      <th>sensor_measurement_17</th>
      <td>0.40</td>
      <td>1.50</td>
      <td>0.07</td>
      <td>0.26</td>
      <td>0.54</td>
      <td>1.30</td>
      <td>1.72</td>
      <td>0.00</td>
      <td>5.62</td>
      <td>&lt;0.005</td>
      <td>25.66</td>
    </tr>
    <tr>
      <th>sensor_measurement_20</th>
      <td>-1.43</td>
      <td>0.24</td>
      <td>0.70</td>
      <td>-2.80</td>
      <td>-0.05</td>
      <td>0.06</td>
      <td>0.95</td>
      <td>0.00</td>
      <td>-2.04</td>
      <td>0.04</td>
      <td>4.59</td>
    </tr>
    <tr>
      <th>sensor_measurement_21</th>
      <td>-3.36</td>
      <td>0.03</td>
      <td>1.19</td>
      <td>-5.69</td>
      <td>-1.04</td>
      <td>0.00</td>
      <td>0.35</td>
      <td>0.00</td>
      <td>-2.84</td>
      <td>&lt;0.005</td>
      <td>7.77</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Partial AIC</th>
      <td>1086.96</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>1328.00 on 18 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>898.23</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_129_1.png)
    


### Predict And Evaluate


```python
RUL_timevary_df_rcensored = RUL_timevary_df[['machine_name', 'cycle', 'operational_setting_1',
       'operational_setting_2', 'operational_setting_3',
        'sensor_measurement_02',
       'sensor_measurement_03', 'sensor_measurement_04',
      
       'sensor_measurement_07', 'sensor_measurement_08',
       'sensor_measurement_09', 'sensor_measurement_10',
       'sensor_measurement_11', 'sensor_measurement_12',
       'sensor_measurement_13', 'sensor_measurement_14',
       'sensor_measurement_15', 
       'sensor_measurement_17',  'sensor_measurement_20',
       'sensor_measurement_21', 'start', 'status', "RUL"]].copy()[RUL_timevary_df['cycle'] <= 215]

RUL_timevary_df_rcensored
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_1</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_07</th>
      <th>sensor_measurement_08</th>
      <th>...</th>
      <th>sensor_measurement_12</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
      <th>start</th>
      <th>status</th>
      <th>RUL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>34.9983</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>555.32</td>
      <td>1358.61</td>
      <td>1137.23</td>
      <td>194.64</td>
      <td>2222.65</td>
      <td>...</td>
      <td>183.06</td>
      <td>2387.72</td>
      <td>8048.56</td>
      <td>9.3461</td>
      <td>334</td>
      <td>14.73</td>
      <td>8.8071</td>
      <td>0</td>
      <td>False</td>
      <td>148</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>41.9982</td>
      <td>0.8408</td>
      <td>100.0</td>
      <td>549.90</td>
      <td>1353.22</td>
      <td>1125.78</td>
      <td>138.51</td>
      <td>2211.57</td>
      <td>...</td>
      <td>130.42</td>
      <td>2387.66</td>
      <td>8072.30</td>
      <td>9.3774</td>
      <td>330</td>
      <td>10.41</td>
      <td>6.2665</td>
      <td>1</td>
      <td>False</td>
      <td>147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>24.9988</td>
      <td>0.6218</td>
      <td>60.0</td>
      <td>537.31</td>
      <td>1256.76</td>
      <td>1047.45</td>
      <td>175.71</td>
      <td>1915.11</td>
      <td>...</td>
      <td>164.22</td>
      <td>2028.03</td>
      <td>7864.87</td>
      <td>10.8941</td>
      <td>309</td>
      <td>14.08</td>
      <td>8.6723</td>
      <td>2</td>
      <td>False</td>
      <td>146</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>42.0077</td>
      <td>0.8416</td>
      <td>100.0</td>
      <td>549.51</td>
      <td>1354.03</td>
      <td>1126.38</td>
      <td>138.46</td>
      <td>2211.58</td>
      <td>...</td>
      <td>130.72</td>
      <td>2387.61</td>
      <td>8068.66</td>
      <td>9.3528</td>
      <td>329</td>
      <td>10.59</td>
      <td>6.4701</td>
      <td>3</td>
      <td>False</td>
      <td>145</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>25.0005</td>
      <td>0.6203</td>
      <td>60.0</td>
      <td>537.07</td>
      <td>1257.71</td>
      <td>1047.93</td>
      <td>175.05</td>
      <td>1915.10</td>
      <td>...</td>
      <td>164.31</td>
      <td>2028.00</td>
      <td>7861.23</td>
      <td>10.8963</td>
      <td>309</td>
      <td>14.13</td>
      <td>8.5286</td>
      <td>4</td>
      <td>False</td>
      <td>144</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>53653</th>
      <td>260</td>
      <td>211</td>
      <td>9.9985</td>
      <td>0.2500</td>
      <td>100.0</td>
      <td>604.58</td>
      <td>1510.23</td>
      <td>1308.38</td>
      <td>394.46</td>
      <td>2318.91</td>
      <td>...</td>
      <td>371.71</td>
      <td>2388.11</td>
      <td>8145.03</td>
      <td>8.6239</td>
      <td>368</td>
      <td>28.69</td>
      <td>17.1938</td>
      <td>210</td>
      <td>False</td>
      <td>105</td>
    </tr>
    <tr>
      <th>53654</th>
      <td>260</td>
      <td>212</td>
      <td>35.0062</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>555.67</td>
      <td>1366.07</td>
      <td>1132.65</td>
      <td>194.45</td>
      <td>2223.23</td>
      <td>...</td>
      <td>183.63</td>
      <td>2388.42</td>
      <td>8083.57</td>
      <td>9.2709</td>
      <td>335</td>
      <td>14.82</td>
      <td>8.8813</td>
      <td>211</td>
      <td>False</td>
      <td>104</td>
    </tr>
    <tr>
      <th>53655</th>
      <td>260</td>
      <td>213</td>
      <td>42.0066</td>
      <td>0.8404</td>
      <td>100.0</td>
      <td>549.48</td>
      <td>1353.35</td>
      <td>1123.50</td>
      <td>138.63</td>
      <td>2212.20</td>
      <td>...</td>
      <td>130.71</td>
      <td>2388.38</td>
      <td>8101.10</td>
      <td>9.3430</td>
      <td>331</td>
      <td>10.65</td>
      <td>6.3974</td>
      <td>212</td>
      <td>False</td>
      <td>103</td>
    </tr>
    <tr>
      <th>53656</th>
      <td>260</td>
      <td>214</td>
      <td>0.0026</td>
      <td>0.0001</td>
      <td>100.0</td>
      <td>643.07</td>
      <td>1595.53</td>
      <td>1409.50</td>
      <td>554.03</td>
      <td>2388.08</td>
      <td>...</td>
      <td>521.82</td>
      <td>2388.08</td>
      <td>8159.58</td>
      <td>8.4068</td>
      <td>394</td>
      <td>38.76</td>
      <td>23.2706</td>
      <td>213</td>
      <td>False</td>
      <td>102</td>
    </tr>
    <tr>
      <th>53657</th>
      <td>260</td>
      <td>215</td>
      <td>34.9994</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>555.75</td>
      <td>1363.27</td>
      <td>1132.58</td>
      <td>194.05</td>
      <td>2223.26</td>
      <td>...</td>
      <td>183.14</td>
      <td>2388.43</td>
      <td>8088.93</td>
      <td>9.3100</td>
      <td>335</td>
      <td>14.90</td>
      <td>8.8744</td>
      <td>214</td>
      <td>False</td>
      <td>101</td>
    </tr>
  </tbody>
</table>
<p>49862 rows × 23 columns</p>
</div>




```python
preds_df = RUL_timevary_df_rcensored.groupby("machine_name").last()
preds_df = preds_df[preds_df.status == False].copy()

preds_df_wo_RUL = preds_df.drop(['RUL'], axis=1)
```


```python

predictions = pd.DataFrame(ctv.predict_log_partial_hazard(preds_df_wo_RUL), columns=['predictions'])
predictions.index = preds_df.index

predictions['RUL'] = preds_df['RUL']
predictions.head(10)
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
      <th>predictions</th>
      <th>RUL</th>
    </tr>
    <tr>
      <th>machine_name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>4.822688</td>
      <td>54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.431523</td>
      <td>20</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.342774</td>
      <td>56</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.713341</td>
      <td>34</td>
    </tr>
    <tr>
      <th>13</th>
      <td>7.702615</td>
      <td>12</td>
    </tr>
    <tr>
      <th>31</th>
      <td>-2.501934</td>
      <td>128</td>
    </tr>
    <tr>
      <th>32</th>
      <td>6.108181</td>
      <td>66</td>
    </tr>
    <tr>
      <th>41</th>
      <td>4.688602</td>
      <td>47</td>
    </tr>
    <tr>
      <th>42</th>
      <td>11.195667</td>
      <td>6</td>
    </tr>
    <tr>
      <th>47</th>
      <td>5.124843</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plt.xlabel callable bugfix
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)

plt.figure(figsize=(15,5))
plt.plot(predictions['RUL'], predictions['predictions'], '.b')
xlim = plt.gca().get_xlim()
plt.xlim(xlim[1], xlim[0])
plt.xlabel('RUL')
plt.ylabel('log_partial_hazard')
plt.title('log_partial_hazard vs. Remaining useful Lifetime of all engines still working after cycle 215')
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_134_0.png)
    


### Predicting partial hazards


```python
partial_hazard_df = RUL_timevary_df[['machine_name', 'cycle', 
       'operational_setting_2', 'operational_setting_3',
        'sensor_measurement_02',
       'sensor_measurement_03', 'sensor_measurement_04',
       'sensor_measurement_07', 'sensor_measurement_08',
       'sensor_measurement_09', 'sensor_measurement_10',
       'sensor_measurement_11', 'sensor_measurement_12',
       'sensor_measurement_13', 'sensor_measurement_14',
       'sensor_measurement_15', 
       'sensor_measurement_17',  'sensor_measurement_20',
       'sensor_measurement_21',  'start', 'status', "RUL"]].copy()

partial_hazard_df_test = RUL_timevary_df_test[['machine_name', 'cycle', 
       'operational_setting_2', 'operational_setting_3',
        'sensor_measurement_02',
       'sensor_measurement_03', 'sensor_measurement_04',
       'sensor_measurement_07', 'sensor_measurement_08',
       'sensor_measurement_09', 'sensor_measurement_10',
       'sensor_measurement_11', 'sensor_measurement_12',
       'sensor_measurement_13', 'sensor_measurement_14',
       'sensor_measurement_15', 
       'sensor_measurement_17',  'sensor_measurement_20',
       'sensor_measurement_21',  'start', 'status', "RUL"]].copy()

partial_hazard_df_wo_RUL =  partial_hazard_df.drop(['RUL'], axis=1)
partial_hazard_df_test_wo_RUL =  partial_hazard_df_test.drop(['RUL'], axis=1)
partial_hazard_df_wo_RUL.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 53759 entries, 0 to 53758
    Data columns (total 21 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   machine_name           53759 non-null  int64  
     1   cycle                  53759 non-null  int64  
     2   operational_setting_2  53759 non-null  float64
     3   operational_setting_3  53759 non-null  float64
     4   sensor_measurement_02  53759 non-null  float64
     5   sensor_measurement_03  53759 non-null  float64
     6   sensor_measurement_04  53759 non-null  float64
     7   sensor_measurement_07  53759 non-null  float64
     8   sensor_measurement_08  53759 non-null  float64
     9   sensor_measurement_09  53759 non-null  float64
     10  sensor_measurement_10  53759 non-null  float64
     11  sensor_measurement_11  53759 non-null  float64
     12  sensor_measurement_12  53759 non-null  float64
     13  sensor_measurement_13  53759 non-null  float64
     14  sensor_measurement_14  53759 non-null  float64
     15  sensor_measurement_15  53759 non-null  float64
     16  sensor_measurement_17  53759 non-null  int64  
     17  sensor_measurement_20  53759 non-null  float64
     18  sensor_measurement_21  53759 non-null  float64
     19  start                  53759 non-null  int64  
     20  status                 53759 non-null  bool   
    dtypes: bool(1), float64(16), int64(4)
    memory usage: 8.7 MB
    


```python
ctv_all = CoxTimeVaryingFitter()
ctv_all.fit(partial_hazard_df_wo_RUL, id_col="machine_name", event_col='status', 
        start_col='start', stop_col='cycle', show_progress=True, step_size=1)
```

    Iteration 45: norm_delta = 0.00315, step_size = 1.00000, ll = -537.06675, newton_decrement = 0.00000, seconds_since_start = 2.8Convergence completed after 45 iterations.
    




    <lifelines.CoxTimeVaryingFitter: fitted with 53759 periods, 260 subjects, 260 events>




```python
partial_hazard_df['log_partial_hazard'] = ctv_all.predict_log_partial_hazard(partial_hazard_df_wo_RUL)
partial_hazard_df_test['log_partial_hazard'] = ctv_all.predict_log_partial_hazard(partial_hazard_df_test_wo_RUL)
partial_hazard_df[-220:-1]
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
      <th>machine_name</th>
      <th>cycle</th>
      <th>operational_setting_2</th>
      <th>operational_setting_3</th>
      <th>sensor_measurement_02</th>
      <th>sensor_measurement_03</th>
      <th>sensor_measurement_04</th>
      <th>sensor_measurement_07</th>
      <th>sensor_measurement_08</th>
      <th>sensor_measurement_09</th>
      <th>...</th>
      <th>sensor_measurement_13</th>
      <th>sensor_measurement_14</th>
      <th>sensor_measurement_15</th>
      <th>sensor_measurement_17</th>
      <th>sensor_measurement_20</th>
      <th>sensor_measurement_21</th>
      <th>start</th>
      <th>status</th>
      <th>RUL</th>
      <th>log_partial_hazard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53539</th>
      <td>260</td>
      <td>97</td>
      <td>0.6200</td>
      <td>60.0</td>
      <td>537.22</td>
      <td>1255.25</td>
      <td>1046.52</td>
      <td>176.24</td>
      <td>1915.32</td>
      <td>8008.15</td>
      <td>...</td>
      <td>2028.26</td>
      <td>7874.49</td>
      <td>10.8557</td>
      <td>306</td>
      <td>14.42</td>
      <td>8.5486</td>
      <td>96</td>
      <td>False</td>
      <td>219</td>
      <td>-2.405619</td>
    </tr>
    <tr>
      <th>53540</th>
      <td>260</td>
      <td>98</td>
      <td>0.0006</td>
      <td>100.0</td>
      <td>642.20</td>
      <td>1586.33</td>
      <td>1395.17</td>
      <td>553.74</td>
      <td>2388.05</td>
      <td>9058.69</td>
      <td>...</td>
      <td>2388.10</td>
      <td>8135.97</td>
      <td>8.3956</td>
      <td>392</td>
      <td>38.94</td>
      <td>23.4231</td>
      <td>97</td>
      <td>False</td>
      <td>218</td>
      <td>-7.915739</td>
    </tr>
    <tr>
      <th>53541</th>
      <td>260</td>
      <td>99</td>
      <td>0.0020</td>
      <td>100.0</td>
      <td>642.39</td>
      <td>1593.33</td>
      <td>1400.30</td>
      <td>554.78</td>
      <td>2388.07</td>
      <td>9054.35</td>
      <td>...</td>
      <td>2388.07</td>
      <td>8137.09</td>
      <td>8.4222</td>
      <td>392</td>
      <td>38.97</td>
      <td>23.3577</td>
      <td>98</td>
      <td>False</td>
      <td>217</td>
      <td>-5.113721</td>
    </tr>
    <tr>
      <th>53542</th>
      <td>260</td>
      <td>100</td>
      <td>0.8400</td>
      <td>100.0</td>
      <td>549.40</td>
      <td>1355.57</td>
      <td>1124.39</td>
      <td>138.06</td>
      <td>2211.87</td>
      <td>8325.03</td>
      <td>...</td>
      <td>2388.03</td>
      <td>8088.60</td>
      <td>9.3724</td>
      <td>330</td>
      <td>10.72</td>
      <td>6.3828</td>
      <td>99</td>
      <td>False</td>
      <td>216</td>
      <td>-2.618363</td>
    </tr>
    <tr>
      <th>53543</th>
      <td>260</td>
      <td>101</td>
      <td>0.6200</td>
      <td>60.0</td>
      <td>536.92</td>
      <td>1262.05</td>
      <td>1042.80</td>
      <td>175.73</td>
      <td>1915.34</td>
      <td>8010.41</td>
      <td>...</td>
      <td>2028.23</td>
      <td>7873.90</td>
      <td>10.8736</td>
      <td>307</td>
      <td>14.17</td>
      <td>8.5988</td>
      <td>100</td>
      <td>False</td>
      <td>215</td>
      <td>-1.891851</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>53753</th>
      <td>260</td>
      <td>311</td>
      <td>0.0010</td>
      <td>100.0</td>
      <td>644.15</td>
      <td>1604.81</td>
      <td>1427.53</td>
      <td>552.02</td>
      <td>2388.13</td>
      <td>9184.23</td>
      <td>...</td>
      <td>2388.14</td>
      <td>8246.20</td>
      <td>8.5303</td>
      <td>396</td>
      <td>38.46</td>
      <td>23.1252</td>
      <td>310</td>
      <td>False</td>
      <td>5</td>
      <td>11.023303</td>
    </tr>
    <tr>
      <th>53754</th>
      <td>260</td>
      <td>312</td>
      <td>0.7000</td>
      <td>100.0</td>
      <td>608.79</td>
      <td>1495.60</td>
      <td>1269.51</td>
      <td>334.00</td>
      <td>2324.92</td>
      <td>8852.27</td>
      <td>...</td>
      <td>2389.02</td>
      <td>8169.64</td>
      <td>9.3035</td>
      <td>369</td>
      <td>24.36</td>
      <td>14.5189</td>
      <td>311</td>
      <td>False</td>
      <td>4</td>
      <td>7.773097</td>
    </tr>
    <tr>
      <th>53755</th>
      <td>260</td>
      <td>313</td>
      <td>0.2510</td>
      <td>100.0</td>
      <td>605.81</td>
      <td>1514.32</td>
      <td>1324.12</td>
      <td>392.46</td>
      <td>2319.17</td>
      <td>8915.86</td>
      <td>...</td>
      <td>2388.42</td>
      <td>8245.36</td>
      <td>8.7586</td>
      <td>374</td>
      <td>28.10</td>
      <td>16.9454</td>
      <td>312</td>
      <td>False</td>
      <td>3</td>
      <td>11.300359</td>
    </tr>
    <tr>
      <th>53756</th>
      <td>260</td>
      <td>314</td>
      <td>0.6200</td>
      <td>60.0</td>
      <td>537.48</td>
      <td>1276.24</td>
      <td>1057.92</td>
      <td>175.11</td>
      <td>1917.37</td>
      <td>8114.33</td>
      <td>...</td>
      <td>2030.33</td>
      <td>7971.25</td>
      <td>11.0657</td>
      <td>310</td>
      <td>14.19</td>
      <td>8.5503</td>
      <td>313</td>
      <td>False</td>
      <td>2</td>
      <td>11.386472</td>
    </tr>
    <tr>
      <th>53757</th>
      <td>260</td>
      <td>315</td>
      <td>0.6220</td>
      <td>60.0</td>
      <td>537.84</td>
      <td>1272.95</td>
      <td>1066.30</td>
      <td>174.81</td>
      <td>1917.36</td>
      <td>8112.63</td>
      <td>...</td>
      <td>2030.35</td>
      <td>7972.47</td>
      <td>11.0537</td>
      <td>311</td>
      <td>14.05</td>
      <td>8.3729</td>
      <td>314</td>
      <td>False</td>
      <td>1</td>
      <td>14.245753</td>
    </tr>
  </tbody>
</table>
<p>219 rows × 23 columns</p>
</div>




```python
rul10ph = partial_hazard_df[partial_hazard_df.RUL < 11][['RUL', 'log_partial_hazard']]
rul10ph
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
      <th>RUL</th>
      <th>log_partial_hazard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>10</td>
      <td>9.825637</td>
    </tr>
    <tr>
      <th>139</th>
      <td>9</td>
      <td>8.224007</td>
    </tr>
    <tr>
      <th>140</th>
      <td>8</td>
      <td>10.549260</td>
    </tr>
    <tr>
      <th>141</th>
      <td>7</td>
      <td>9.158902</td>
    </tr>
    <tr>
      <th>142</th>
      <td>6</td>
      <td>7.731988</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53754</th>
      <td>4</td>
      <td>7.773097</td>
    </tr>
    <tr>
      <th>53755</th>
      <td>3</td>
      <td>11.300359</td>
    </tr>
    <tr>
      <th>53756</th>
      <td>2</td>
      <td>11.386472</td>
    </tr>
    <tr>
      <th>53757</th>
      <td>1</td>
      <td>14.245753</td>
    </tr>
    <tr>
      <th>53758</th>
      <td>0</td>
      <td>10.462597</td>
    </tr>
  </tbody>
</table>
<p>2860 rows × 2 columns</p>
</div>




```python
# plt.xlabel callable bugfix
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)

sns.set(rc = {'figure.figsize':(16,6)})
plt.scatter(partial_hazard_df['RUL'], partial_hazard_df['log_partial_hazard'])

plt.ylabel('log_partial_hazard')
plt.xlabel('RUL')
plt.axhline(y=5,ls="--",c=".5")
xlim = plt.gca().get_xlim()
plt.xlim(xlim[1], xlim[0])
plt.title('Train data - RUL vs. log partial hazard')

plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_140_0.png)
    



```python
sns.set(rc = {'figure.figsize':(16,9)})
g = sns.FacetGrid(rul10ph, col="RUL", 
                  #height=12, 
                  #aspect=1.2, 
                  col_wrap=4)
g.map(sns.histplot, "log_partial_hazard")
g.map(plt.axvline,x=10,ls="--",c=".5", color='r',)
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_141_0.png)
    


### Simulate Engine Retirement Heuristic before RUL = 0

#### Training data


```python
%%time
sns.set(rc = {'figure.figsize':(8,5)})

def loghaz_stopheuristic(df, threshold, recurrence_limit, method='consecutive', window=11):
    machinelist = df.machine_name.unique()
    decision_df = pd.DataFrame(index=machinelist)
    df['stop_decision'] = np.nan
    df['stop_decision_when'] = np.nan
    for m in machinelist:
        mask = (df.machine_name == m)
        temp = df[mask & (df.RUL < (window + 1))]
        overcount = 0
        for index, row in enumerate(temp.itertuples(), 1):
            if row.log_partial_hazard > threshold:
                 overcount += 1
            elif method=='consecutive':
                 overcount = 0
            if overcount > recurrence_limit:
                if row.RUL != 0:
                    decision_df.loc[m , 'stop_decision'] = True
                decision_df.loc[m , 'stop_decision_when'] = row.RUL
                #print(f'machine {m} was stopped' )
                break
            
    return df, decision_df
            

result, decisions = loghaz_stopheuristic(partial_hazard_df, threshold=5.5, recurrence_limit=5, method='consecutive', window=16)

print(f'stopped: {decisions.sum()} total: {decisions.shape[0]}')
      
print(f'\nmedian rul: {decisions.stop_decision_when.median()}')
decisions.stop_decision_when.plot(kind='box')
plt.ylabel('RUL')
plt.title('Distribution of engines retired with remaining RUL before failure (RUL = 0)')
plt.show()
decisions.stop_decision_when.plot(kind='hist', bins=35)
#plt.xlabel('RUL')
plt.title('Histogramm of engines retired with remaining RUL')
plt.show()
```

    stopped: stop_decision            260
    stop_decision_when    2415.0
    dtype: object total: 260
    
    median rul: 11.0
    


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_143_1.png)
    



    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_143_2.png)
    


    CPU times: total: 766 ms
    Wall time: 761 ms
    

#### Verifiying with testing data


```python
r, decisions_test = loghaz_stopheuristic(partial_hazard_df_test, threshold=5, recurrence_limit=3, method='consecutive', window=30)

print(f'stopped: {decisions_test.sum()} total: {decisions_test.shape[0]} median rul: {decisions_test.stop_decision_when.median()}')
decisions_test.stop_decision_when.plot(kind='box')
plt.ylabel('RUL')
plt.title('Distribution of engines retired with remaining RUL before failure (RUL = 0)')
plt.show()
decisions_test.stop_decision_when.plot(kind='hist', bins=35)
#plt.xlabel('RUL')
plt.title('Histogramm of engines retired with remaining RUL')
plt.show()
```

    stopped: stop_decision            53
    stop_decision_when    625.0
    dtype: object total: 259 median rul: 10.0
    


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_145_1.png)
    



    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_145_2.png)
    



```python
decisions_test
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
      <th>stop_decision</th>
      <th>stop_decision_when</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>255</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>256</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>257</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>258</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>259</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>259 rows × 2 columns</p>
</div>




```python
# plt.xlabel callable bugfix
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)

sns.set(rc = {'figure.figsize':(16,11)})
plt.subplot(2,1,1)
#plt.figsize=(15,11)
plt.scatter(partial_hazard_df['RUL'], partial_hazard_df['log_partial_hazard'])

plt.ylabel('log_partial_hazard')
plt.xlabel('RUL')
plt.axhline(y=5,ls="--",c=".5")
xlim = plt.gca().get_xlim()
plt.xlim(xlim[1], xlim[0])
plt.title('Train data - RUL vs. log partial hazard')

plt.subplot(2,1,2)
plt.scatter(partial_hazard_df_test['RUL'], partial_hazard_df_test['log_partial_hazard'], color='g' )

xlim = plt.gca().get_xlim()
plt.xlim(xlim[1], xlim[0])

plt.ylabel('log_partial_hazard')
plt.xlabel('RUL')
plt.title('Test data - RUL vs. log partial hazard')
plt.show()
```


    
![png](survival-analysis-for-predictive-maintenance_files/survival-analysis-for-predictive-maintenance_147_0.png)
    


The CoxTimeVaryingFitted model was trained soley with training data of course. It performs well predicting log partial hazard values on the very same machines and is able to distinguish a nearing failure towards RUL=0 with virtually all log partial hazard values above a threshold of roundabout 5 pretty well from cycles that are far from end of RUL.

Predicting the same on the test data is however sobering as the model is seeing also elevated log partial hazard values it more often than not cannot distinguish near RUL zero engines from others. The model does not generalize well enough and is obviously overfitted. We have come to see the limitations of the CoxTimeVaryingFitting method as means of survival analysis and would recommend reapproach this problem with another algorithm and problem framing.

Update: Following up, I have reapproached this predictive maintenance problem again as a time series window classification problem, implemented as Long Short Term Memory (LSTM) type of recurrent neural network (RNN) suited for time series sequence data and its long-range dependencies. Please see here in my projects section: [Predictive Maintenance ~ failure prediction with binary classification implemented as Gated Recurrent Units Recurrent Neural Network](https://dataninjato.github.io/projects/7-predictive-maintenance-failure-prediction-with-binary-classification-implemented-as-gated-recurrent-units-recurrent-neural-network)

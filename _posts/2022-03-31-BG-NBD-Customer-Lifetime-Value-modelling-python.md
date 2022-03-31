---
title: Non contractual Customer Lifetime Value estimated probabilistically with the Beta Geometric/Negative Binomial Distribution (BG/NBD) Model
tags: [Customer Lifetime Value, Python, Beta Geometric/Negative Binomial Distribution (BG/NBD), Customer Penetration, Purchase Frequency]
style: border
color: danger
description: We assume an online service business where customers/clients continously purchase our services. For such a service business we generate our customer transactions ourselves instead of using once again one of the few available public Datasets. We pick a Gamma distribution for the tenure of our ...
---

We assume an online service business where customers/clients continously purchase our services.

For such a service business we generate our customer transactions ourselves instead of using once again one of the few available public Datasets.

We pick a Gamma distribution for the tenure of our ```10000``` customers with ```shape, scale = 4, 11```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set(rc={"figure.figsize":(10, 8)})

rng = np.random.default_rng(2022)

shape, scale = 4, 11
tenure = rng.gamma(shape, scale, 10000) *2
plt.hist(tenure, density=True, edgecolor='black', bins=50)
plt.title('Histogramm - Customer Tenure')
plt.xlabel('Tenure')
plt.ylabel('Count')
plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_2_0.png)
    


### Using the Beta Geometric/Negative Binomial Distribution (BG/NBD) Model

The BG/NBD is based on the Pareto/NBD model. Unlike to the Pareto/NBD model, the betageometric (BG) model assume that customer die and drop out immideatly after a transaction, whereas Pareto models a probability that dropout with which customers dropout can occurr anytime.


#### what else is assumed?

1. While active, the number of transactions made
   by a customer follows a Poisson process with transaction rate $\lambda$, which essentially means the 
   time between transactions is exponentially distributed.

2. Heterogeneity in $ \lambda $ is gamma distributed 

3. Customer become inactive with probability p after every transaction
   inactive with probability $ p $. Therefore the point in time when the customer dies is distributed (shifted) geometrically      across transactions

4. Heterogeneity in $ p $ follows a beta distribution
5. The transaction rate $ \lambda $ and the dropout probability $ p $ are independent between customers.

   Note:  $ \lambda $ and $ p $ are both unobserved

All customers are assumed to be active customers in this model, so it makes sense to apply it on a cohort of customers who just have made their first purchase.

#### let's generate some daily transactional data for a cohort of fresh customers


```python
%%time
from faker import Faker
import pandas as pd
fake = Faker(['it_IT', 'en_UK', 'fr_FR', 'de_DE', 'uk_UA'])
newcols = fake.simple_profile().keys()

profiles = pd.DataFrame(columns=list(newcols))
for i in range(10000):
    profiles.loc[i] = fake.simple_profile()

profiles['customer_id'] = profiles.index   
first_column = profiles.pop('customer_id')
profiles.insert(0, 'customer_id', first_column)
profiles.head()
```

    CPU times: total: 17.2 s
    Wall time: 17.3 s
    




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
      <th>customer_id</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>abramstey</td>
      <td>Nuran Trupp-Lachmann</td>
      <td>F</td>
      <td>Ritterallee 5/3\n23598 Staffelstein</td>
      <td>babett14@hotmail.de</td>
      <td>1952-02-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>pichonalex</td>
      <td>Marcel Leroy</td>
      <td>M</td>
      <td>69, rue de Ollivier\n25378 GoncalvesBourg</td>
      <td>francoismartin@club-internet.fr</td>
      <td>1966-12-13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>le-gallemmanuelle</td>
      <td>Monique Roche</td>
      <td>F</td>
      <td>85, avenue de Thierry\n67649 Maillet</td>
      <td>madeleineleroy@dbmail.com</td>
      <td>1945-04-22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>jakob73</td>
      <td>Nuray Martin B.Sc.</td>
      <td>F</td>
      <td>Holtallee 7\n19875 Ahaus</td>
      <td>hgeisler@yahoo.de</td>
      <td>1909-09-26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>shvachkanazar</td>
      <td>Тетяна Дергач</td>
      <td>F</td>
      <td>набережна Лемківська, 2, селище Марʼяна, 101472</td>
      <td>leontii90@email.ua</td>
      <td>1936-08-12</td>
    </tr>
  </tbody>
</table>
</div>




```python
# model params rounded from CDNOW sample in the paper p. 281
params=dict()

params['r'] = 0.25, 
params['alpha'] = 4.5

params['a'] = 0.8
params['b'] = 2.4

params
```




    {'r': (0.25,), 'alpha': 4.5, 'a': 0.8, 'b': 2.4}




```python
observation_period_end='2021-12-31'
```


```python
%%time
from lifetimes.generate_data import beta_geometric_nbd_model_transactional_data

transactions = beta_geometric_nbd_model_transactional_data(tenure, params['r'], params['alpha'], params['a'], params['b'], 
                                                  observation_period_end=observation_period_end, freq='D', size=10000)
transactions.shape
```

    CPU times: total: 1min 52s
    Wall time: 1min 52s
    




    (29664, 2)




```python
trans_df = transactions.merge(profiles, left_on='customer_id', right_on='customer_id')
```


```python
trans_df.head()
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
      <th>customer_id</th>
      <th>date</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2021-04-20 21:30:50.384044800</td>
      <td>abramstey</td>
      <td>Nuran Trupp-Lachmann</td>
      <td>F</td>
      <td>Ritterallee 5/3\n23598 Staffelstein</td>
      <td>babett14@hotmail.de</td>
      <td>1952-02-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2021-08-26 06:44:27.043411199</td>
      <td>abramstey</td>
      <td>Nuran Trupp-Lachmann</td>
      <td>F</td>
      <td>Ritterallee 5/3\n23598 Staffelstein</td>
      <td>babett14@hotmail.de</td>
      <td>1952-02-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2021-10-21 23:25:51.052166400</td>
      <td>abramstey</td>
      <td>Nuran Trupp-Lachmann</td>
      <td>F</td>
      <td>Ritterallee 5/3\n23598 Staffelstein</td>
      <td>babett14@hotmail.de</td>
      <td>1952-02-10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2021-06-10 07:05:42.827625600</td>
      <td>pichonalex</td>
      <td>Marcel Leroy</td>
      <td>M</td>
      <td>69, rue de Ollivier\n25378 GoncalvesBourg</td>
      <td>francoismartin@club-internet.fr</td>
      <td>1966-12-13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2021-12-11 21:52:01.636838400</td>
      <td>pichonalex</td>
      <td>Marcel Leroy</td>
      <td>M</td>
      <td>69, rue de Ollivier\n25378 GoncalvesBourg</td>
      <td>francoismartin@club-internet.fr</td>
      <td>1966-12-13</td>
    </tr>
  </tbody>
</table>
</div>



Aggregate the summary data analog to RFM segmentation (Recency, Frequency, Monetary) from the transactional data just generated


```python
from lifetimes.utils import summary_data_from_transaction_data

summary = summary_data_from_transaction_data(trans_df, 'customer_id', 'date', observation_period_end=observation_period_end)
```


```python
summary = pd.concat([profiles, summary], axis=1)
summary
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
      <th>customer_id</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
      <th>frequency</th>
      <th>recency</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>abramstey</td>
      <td>Nuran Trupp-Lachmann</td>
      <td>F</td>
      <td>Ritterallee 5/3\n23598 Staffelstein</td>
      <td>babett14@hotmail.de</td>
      <td>1952-02-10</td>
      <td>2.0</td>
      <td>184.0</td>
      <td>255.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>pichonalex</td>
      <td>Marcel Leroy</td>
      <td>M</td>
      <td>69, rue de Ollivier\n25378 GoncalvesBourg</td>
      <td>francoismartin@club-internet.fr</td>
      <td>1966-12-13</td>
      <td>1.0</td>
      <td>184.0</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>le-gallemmanuelle</td>
      <td>Monique Roche</td>
      <td>F</td>
      <td>85, avenue de Thierry\n67649 Maillet</td>
      <td>madeleineleroy@dbmail.com</td>
      <td>1945-04-22</td>
      <td>3.0</td>
      <td>98.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>jakob73</td>
      <td>Nuray Martin B.Sc.</td>
      <td>F</td>
      <td>Holtallee 7\n19875 Ahaus</td>
      <td>hgeisler@yahoo.de</td>
      <td>1909-09-26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>shvachkanazar</td>
      <td>Тетяна Дергач</td>
      <td>F</td>
      <td>набережна Лемківська, 2, селище Марʼяна, 101472</td>
      <td>leontii90@email.ua</td>
      <td>1936-08-12</td>
      <td>1.0</td>
      <td>45.0</td>
      <td>109.0</td>
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
      <th>9995</th>
      <td>9995</td>
      <td>iarynahavrylyshyn</td>
      <td>Олег Рудько</td>
      <td>M</td>
      <td>набережна Маркіяна Шашкевича, 517, хутір Данил...</td>
      <td>khavrylenko@ukr.net</td>
      <td>1988-07-07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>9996</td>
      <td>gisbertriehl</td>
      <td>Claire Koch-Anders</td>
      <td>F</td>
      <td>Zorbachring 7\n70870 Burgdorf</td>
      <td>tomas93@hotmail.de</td>
      <td>2014-08-13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>9997</td>
      <td>marta63</td>
      <td>Остап Ейбоженко</td>
      <td>M</td>
      <td>вулиця Бруно Шульца, 915, хутір Ганна, 76375</td>
      <td>vdovychenkobohuslav@meta.ua</td>
      <td>1976-07-24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>9998</td>
      <td>opowell</td>
      <td>Chelsea Poole</td>
      <td>F</td>
      <td>Flat 82\nBryan passage\nNorth Luke\nWF1N 0AL</td>
      <td>rsmart@hotmail.co.uk</td>
      <td>1916-10-24</td>
      <td>1.0</td>
      <td>47.0</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>9999</td>
      <td>havrylotsymbaliuk</td>
      <td>пан Устим Цибуленко</td>
      <td>M</td>
      <td>вулиця Василя Симоненка, 628, село Леон, 85627</td>
      <td>venedykt92@ukr.net</td>
      <td>1975-02-06</td>
      <td>2.0</td>
      <td>44.0</td>
      <td>75.0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 10 columns</p>
</div>




```python
"""from lifetimes.generate_data import beta_geometric_nbd_model
#lifetimes.generate_data.beta_geometric_nbd_model(T, r, alpha, a, b, size=1)
#Generate artificial data according to the BG/NBD model.

df = beta_geometric_nbd_model(tenure, params['r'], params['alpha'], params['a'], params['b'], size=10000)
"""
"""Parameters:	
T (array_like) – The length of time observing new customers.
alpha, a, b (r,) – Parameters in the model. See [1]_
size (int, optional) – The number of customers to generate
Returns:	
DataFrame – With index as customer_ids and the following columns: ‘frequency’, ‘recency’, ‘T’, ‘lambda’, ‘p’, ‘alive’, ‘customer_id’"""
```




    'Parameters:\t\nT (array_like) – The length of time observing new customers.\nalpha, a, b (r,) – Parameters in the model. See [1]_\nsize (int, optional) – The number of customers to generate\nReturns:\t\nDataFrame – With index as customer_ids and the following columns: ‘frequency’, ‘recency’, ‘T’, ‘lambda’, ‘p’, ‘alive’, ‘customer_id’'




```python
from lifetimes import BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])
bgf.summary
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
      <th>se(coef)</th>
      <th>lower 95% bound</th>
      <th>upper 95% bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>r</th>
      <td>0.289300</td>
      <td>0.005818</td>
      <td>0.277897</td>
      <td>0.300704</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>6.476316</td>
      <td>0.213714</td>
      <td>6.057437</td>
      <td>6.895194</td>
    </tr>
    <tr>
      <th>a</th>
      <td>0.790777</td>
      <td>0.053952</td>
      <td>0.685031</td>
      <td>0.896524</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.530743</td>
      <td>0.231390</td>
      <td>2.077220</td>
      <td>2.984267</td>
    </tr>
  </tbody>
</table>
</div>




```python
from lifetimes.plotting import plot_frequency_recency_matrix
sns.set(rc={"figure.figsize":(10, 8)})
sns.set_style('dark')
plot_frequency_recency_matrix(bgf, cmap='plasma')
plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_17_0.png)
    



```python
from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf, cmap='plasma')
plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_18_0.png)
    


#### Customer ranking

Let us identify the customers with Top 5 expected purchases within next seven days (t=7) based on their transaction history.


```python
t = 7
ppcolname = 'predicted_purchases_' + str(t)
summary[ppcolname] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'],summary['recency'], summary['T'])
summary.sort_values(by=ppcolname, ascending=False).head(5)
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
      <th>customer_id</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
      <th>frequency</th>
      <th>recency</th>
      <th>T</th>
      <th>predicted_purchases_14</th>
      <th>predicted_purchases_30</th>
      <th>predicted_purchases_7</th>
      <th>monetary_value_average</th>
      <th>Expected_cond_average_revenue</th>
      <th>error_rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2910</th>
      <td>2910</td>
      <td>carlypalmer</td>
      <td>Mrs. Joan Heath</td>
      <td>M</td>
      <td>Studio 41t\nAbigail port\nEast Jayne\nPO5X 1DL</td>
      <td>sally36@yahoo.co.uk</td>
      <td>1908-01-15</td>
      <td>33.0</td>
      <td>55.0</td>
      <td>55.0</td>
      <td>6.851758</td>
      <td>14.839702</td>
      <td>3.557546</td>
      <td>38.734677</td>
      <td>38.622364</td>
      <td>-0.112313</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>3129</td>
      <td>jsontag</td>
      <td>Zeynep Steckel</td>
      <td>F</td>
      <td>Biengasse 720\n46808 Ansbach</td>
      <td>peukertmeike@aol.de</td>
      <td>1976-05-28</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>6.324447</td>
      <td>13.947335</td>
      <td>3.344909</td>
      <td>31.671466</td>
      <td>31.566594</td>
      <td>-0.104872</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>1677</td>
      <td>bohodarokhrimenko</td>
      <td>Болеслав Дубас</td>
      <td>M</td>
      <td>вулиця Шота Руставелі, 248, місто Лариса, 11399</td>
      <td>artemtymchuk@gmail.com</td>
      <td>1974-01-16</td>
      <td>19.0</td>
      <td>34.0</td>
      <td>34.0</td>
      <td>5.749406</td>
      <td>12.524436</td>
      <td>3.029490</td>
      <td>16.370829</td>
      <td>16.441952</td>
      <td>0.071123</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>3149</td>
      <td>ujones</td>
      <td>Kieran Jennings</td>
      <td>F</td>
      <td>Flat 3\nRoger port\nKieranton\nSR3 0HP</td>
      <td>abigailphillips@gmail.com</td>
      <td>2012-02-22</td>
      <td>34.0</td>
      <td>71.0</td>
      <td>72.0</td>
      <td>5.554099</td>
      <td>12.087839</td>
      <td>2.862532</td>
      <td>7.135654</td>
      <td>7.237132</td>
      <td>0.101478</td>
    </tr>
    <tr>
      <th>7500</th>
      <td>7500</td>
      <td>elombardi</td>
      <td>Dott. Melania Toldo</td>
      <td>F</td>
      <td>Incrocio Bragaglia 3\nCostanzi sardo, 94137 Tr...</td>
      <td>cgagliano@tim.it</td>
      <td>2016-02-24</td>
      <td>23.0</td>
      <td>47.0</td>
      <td>47.0</td>
      <td>5.412396</td>
      <td>11.756576</td>
      <td>2.821701</td>
      <td>12.182490</td>
      <td>12.282481</td>
      <td>0.099991</td>
    </tr>
  </tbody>
</table>
</div>




```python
t = 30
ppcolname = 'predicted_purchases_' + str(t)
summary[ppcolname] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'],summary['recency'], summary['T'])
sorted_summary = summary.sort_values(by=ppcolname, ascending=False)
sorted_summary.head()
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
      <th>customer_id</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
      <th>frequency</th>
      <th>recency</th>
      <th>T</th>
      <th>predicted_purchases_14</th>
      <th>predicted_purchases_30</th>
      <th>predicted_purchases_7</th>
      <th>monetary_value_average</th>
      <th>Expected_cond_average_revenue</th>
      <th>error_rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2910</th>
      <td>2910</td>
      <td>carlypalmer</td>
      <td>Mrs. Joan Heath</td>
      <td>M</td>
      <td>Studio 41t\nAbigail port\nEast Jayne\nPO5X 1DL</td>
      <td>sally36@yahoo.co.uk</td>
      <td>1908-01-15</td>
      <td>33.0</td>
      <td>55.0</td>
      <td>55.0</td>
      <td>6.851758</td>
      <td>13.594318</td>
      <td>3.557546</td>
      <td>38.734677</td>
      <td>38.622364</td>
      <td>-0.112313</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>3129</td>
      <td>jsontag</td>
      <td>Zeynep Steckel</td>
      <td>F</td>
      <td>Biengasse 720\n46808 Ansbach</td>
      <td>peukertmeike@aol.de</td>
      <td>1976-05-28</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>6.324447</td>
      <td>12.149527</td>
      <td>3.344909</td>
      <td>31.671466</td>
      <td>31.566594</td>
      <td>-0.104872</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>3149</td>
      <td>ujones</td>
      <td>Kieran Jennings</td>
      <td>F</td>
      <td>Flat 3\nRoger port\nKieranton\nSR3 0HP</td>
      <td>abigailphillips@gmail.com</td>
      <td>2012-02-22</td>
      <td>34.0</td>
      <td>71.0</td>
      <td>72.0</td>
      <td>5.554099</td>
      <td>11.173048</td>
      <td>2.862532</td>
      <td>7.135654</td>
      <td>7.237132</td>
      <td>0.101478</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>1677</td>
      <td>bohodarokhrimenko</td>
      <td>Болеслав Дубас</td>
      <td>M</td>
      <td>вулиця Шота Руставелі, 248, місто Лариса, 11399</td>
      <td>artemtymchuk@gmail.com</td>
      <td>1974-01-16</td>
      <td>19.0</td>
      <td>34.0</td>
      <td>34.0</td>
      <td>5.749406</td>
      <td>11.114488</td>
      <td>3.029490</td>
      <td>16.370829</td>
      <td>16.441952</td>
      <td>0.071123</td>
    </tr>
    <tr>
      <th>7500</th>
      <td>7500</td>
      <td>elombardi</td>
      <td>Dott. Melania Toldo</td>
      <td>F</td>
      <td>Incrocio Bragaglia 3\nCostanzi sardo, 94137 Tr...</td>
      <td>cgagliano@tim.it</td>
      <td>2016-02-24</td>
      <td>23.0</td>
      <td>47.0</td>
      <td>47.0</td>
      <td>5.412396</td>
      <td>10.659459</td>
      <td>2.821701</td>
      <td>12.182490</td>
      <td>12.282481</td>
      <td>0.099991</td>
    </tr>
  </tbody>
</table>
</div>



### Model fit assessment


```python
from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)
plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_23_0.png)
    



```python
calibration_period_end='2021-10-10'
```


```python
from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(trans_df, 'customer_id', 'date',
                                        calibration_period_end=calibration_period_end,
                                        observation_period_end=observation_period_end )
summary_cal_holdout.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4998 entries, 0 to 9998
    Data columns (total 5 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   frequency_cal      4998 non-null   float64
     1   recency_cal        4998 non-null   float64
     2   T_cal              4998 non-null   float64
     3   frequency_holdout  4998 non-null   float64
     4   duration_holdout   4998 non-null   float64
    dtypes: float64(5)
    memory usage: 363.3 KB
    


```python
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)
plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_26_0.png)
    



```python
bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_cumulative_transactions(bgf, trans_df, 'date', 'customer_id', 280, 140);
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_27_0.png)
    



```python
plot_incremental_transactions(bgf, trans_df, 'date', 'customer_id',280, 140);
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_28_0.png)
    



```python
customer_X = sorted_summary[200:201]
customer_X
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
      <th>customer_id</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
      <th>frequency</th>
      <th>recency</th>
      <th>T</th>
      <th>predicted_purchases_14</th>
      <th>predicted_purchases_30</th>
      <th>predicted_purchases_7</th>
      <th>monetary_value_average</th>
      <th>Expected_cond_average_revenue</th>
      <th>error_rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8403</th>
      <td>8403</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
      <td>7.0</td>
      <td>44.0</td>
      <td>46.0</td>
      <td>1.601542</td>
      <td>3.179439</td>
      <td>0.831359</td>
      <td>18.652597</td>
      <td>18.769815</td>
      <td>0.117218</td>
    </tr>
  </tbody>
</table>
</div>




```python
t = 30 # predict number of purchases in next t periods

individual = summary.iloc[customer_X.customer_id]
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])
# 0.0576511
```




    8403    3.374978
    dtype: float64




```python
from lifetimes.plotting import plot_history_alive

days_since_birth = 61 # of this customer X

sp_trans = trans_df.loc[trans_df['customer_id'] == int(customer_X.customer_id)]
plot_history_alive(bgf, days_since_birth, sp_trans, 'date', title='History of P_alive of customer_X='+str(customer_X.username) )
plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_31_0.png)
    



```python
sp_trans
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
      <th>customer_id</th>
      <th>date</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25086</th>
      <td>8403</td>
      <td>2021-11-15 11:51:07.696598400</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
    <tr>
      <th>25087</th>
      <td>8403</td>
      <td>2021-11-16 21:17:22.970515200</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
    <tr>
      <th>25088</th>
      <td>8403</td>
      <td>2021-11-24 04:46:07.064832</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
    <tr>
      <th>25089</th>
      <td>8403</td>
      <td>2021-12-02 19:17:03.481641600</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
    <tr>
      <th>25090</th>
      <td>8403</td>
      <td>2021-12-13 04:24:15.312902400</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
    <tr>
      <th>25091</th>
      <td>8403</td>
      <td>2021-12-15 16:12:24.811142400</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
    <tr>
      <th>25092</th>
      <td>8403</td>
      <td>2021-12-23 23:41:56.826096</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
    <tr>
      <th>25093</th>
      <td>8403</td>
      <td>2021-12-29 12:19:29.010403200</td>
      <td>trubinguenther</td>
      <td>Ing. Hanno Ullmann</td>
      <td>M</td>
      <td>Gudegasse 8/4\n87671 Grevenbroich</td>
      <td>riehlmarlen@yahoo.de</td>
      <td>1909-03-31</td>
    </tr>
  </tbody>
</table>
</div>




```python
no_transactions = trans_df.groupby('customer_id').count().sort_values('date', ascending=False)['date']
no_transactions 
```




    customer_id
    2019    88
    5280    87
    985     75
    7987    66
    7892    57
            ..
    1722     1
    5653     1
    5651     1
    5650     1
    6404     1
    Name: date, Length: 10000, dtype: int64




```python
df3 = summary[summary['frequency']>0]
df3['prob_alive'] = bgf.conditional_probability_alive(df3['frequency'],df3['recency'],df3['T'])
sns.distplot(df3['prob_alive']);
plt.xlim(0,1)
plt.suptitle('Distribution of Alive Probabilities')
plt.show()

df3['churn'] = ['churned' if p < .1 else 'not churned' for p in df3['prob_alive']]
sns.countplot(df3['churn']);
plt.suptitle('Grouping of alive probabilities')
plt.show()

print('Grouping of alive probabilities')
df3['churn'][(df3['prob_alive']>=.1) & (df3['prob_alive']<.2)] = "high risk"
df3['churn'].value_counts()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_34_0.png)
    



    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_34_1.png)
    


    Grouping of alive probabilities
    




    not churned    3841
    churned         772
    high risk       589
    Name: churn, dtype: int64



### Model monetary transaction value data modeled with Gamma-Gamma model


```python
## monetray value data gen

shape, scale = 3, 3.  # mean=4, std=2*sqrt(2)
monetary_value_average = rng.gamma(shape, scale, 10000) * 2
plt.hist(monetary_value_average*2, density=True, edgecolor='black', bins=50)
plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_36_0.png)
    



```python
summary['monetary_value_average'] = transaction_amount

returning_customers_summary = summary[summary['frequency']>0]
```


```python
returning_customers_summary.head()
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
      <th>customer_id</th>
      <th>username</th>
      <th>name</th>
      <th>sex</th>
      <th>address</th>
      <th>mail</th>
      <th>birthdate</th>
      <th>frequency</th>
      <th>recency</th>
      <th>T</th>
      <th>predicted_purchases_14</th>
      <th>predicted_purchases_30</th>
      <th>predicted_purchases_7</th>
      <th>monetary_value_average</th>
      <th>Expected_cond_average_revenue</th>
      <th>error_rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>abramstey</td>
      <td>Nuran Trupp-Lachmann</td>
      <td>F</td>
      <td>Ritterallee 5/3\n23598 Staffelstein</td>
      <td>babett14@hotmail.de</td>
      <td>1952-02-10</td>
      <td>2.0</td>
      <td>184.0</td>
      <td>255.0</td>
      <td>0.082734</td>
      <td>0.174773</td>
      <td>0.041632</td>
      <td>32.613560</td>
      <td>31.565017</td>
      <td>-1.048543</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>pichonalex</td>
      <td>Marcel Leroy</td>
      <td>M</td>
      <td>69, rue de Ollivier\n25378 GoncalvesBourg</td>
      <td>francoismartin@club-internet.fr</td>
      <td>1966-12-13</td>
      <td>1.0</td>
      <td>184.0</td>
      <td>204.0</td>
      <td>0.062412</td>
      <td>0.131735</td>
      <td>0.031418</td>
      <td>16.727437</td>
      <td>17.774840</td>
      <td>1.047403</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>le-gallemmanuelle</td>
      <td>Monique Roche</td>
      <td>F</td>
      <td>85, avenue de Thierry\n67649 Maillet</td>
      <td>madeleineleroy@dbmail.com</td>
      <td>1945-04-22</td>
      <td>3.0</td>
      <td>98.0</td>
      <td>99.0</td>
      <td>0.357496</td>
      <td>0.738755</td>
      <td>0.181780</td>
      <td>8.692225</td>
      <td>9.658282</td>
      <td>0.966057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>shvachkanazar</td>
      <td>Тетяна Дергач</td>
      <td>F</td>
      <td>набережна Лемківська, 2, селище Марʼяна, 101472</td>
      <td>leontii90@email.ua</td>
      <td>1936-08-12</td>
      <td>1.0</td>
      <td>45.0</td>
      <td>109.0</td>
      <td>0.080884</td>
      <td>0.168804</td>
      <td>0.040935</td>
      <td>14.807318</td>
      <td>16.211255</td>
      <td>1.403937</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>petro07</td>
      <td>Алла Рябець</td>
      <td>F</td>
      <td>набережна Дністровська, 843, хутір Ілля, 106205</td>
      <td>sviatoslavadurdynets@ukr.net</td>
      <td>2002-08-12</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>34.0</td>
      <td>0.172044</td>
      <td>0.345416</td>
      <td>0.088817</td>
      <td>25.410872</td>
      <td>24.845904</td>
      <td>-0.564968</td>
    </tr>
  </tbody>
</table>
</div>



Important assumption for the Gamma-Gamma Model: the relationship between the monetary value and the purchase frequency is near zero.

As this is met we can continue to train the model and start analysing.


```python
returning_customers_summary[['monetary_value_average', 'frequency']].corr()
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
      <th>monetary_value_average</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>monetary_value_average</th>
      <td>1.00000</td>
      <td>-0.00149</td>
    </tr>
    <tr>
      <th>frequency</th>
      <td>-0.00149</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0.002)
ggf.fit(returning_customers_summary['frequency'], returning_customers_summary['monetary_value_average'])

```




    <lifetimes.GammaGammaFitter: fitted with 5202 subjects, p: 5.53, q: 2.26, v: 5.10>




```python
ggf.summary
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
      <th>se(coef)</th>
      <th>lower 95% bound</th>
      <th>upper 95% bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>5.532698</td>
      <td>0.115701</td>
      <td>5.305924</td>
      <td>5.759472</td>
    </tr>
    <tr>
      <th>q</th>
      <td>2.261587</td>
      <td>0.044056</td>
      <td>2.175237</td>
      <td>2.347938</td>
    </tr>
    <tr>
      <th>v</th>
      <td>5.100490</td>
      <td>0.125329</td>
      <td>4.854845</td>
      <td>5.346135</td>
    </tr>
  </tbody>
</table>
</div>




```python
summary['Expected_cond_average_revenue'] = ggf.conditional_expected_average_profit(summary['frequency'], summary['monetary_value_average'])
```


```python
summary['Expected_cond_average_revenue'].describe()
```




    count    10000.000000
    mean        20.413614
    std          7.062209
    min          1.499528
    25%         16.409391
    50%         22.368232
    75%         22.368232
    max         75.235044
    Name: Expected_cond_average_revenue, dtype: float64




```python
# MAPE
from sklearn.metrics import mean_absolute_percentage_error

summary["error_rev"] = summary['Expected_cond_average_revenue'] - summary['monetary_value_average']

mape = mean_absolute_percentage_error(summary['Expected_cond_average_revenue'], summary["monetary_value_average"])
print("MAPE of predicted revenues:", f'{mape:.3f}')

```

    MAPE of predicted revenues: 0.244
    


```python
summary['Expected_cond_average_revenue'].head(20)
```




    0     31.565017
    1     17.774840
    2      9.658282
    3     22.368232
    4     16.211255
    5     24.845904
    6     22.368232
    7     22.368232
    8     22.368232
    9      8.200775
    10    23.078353
    11     6.811015
    12     9.628732
    13    22.368232
    14    22.368232
    15    22.368232
    16    19.063189
    17    29.568306
    18    34.173798
    19    12.974695
    Name: Expected_cond_average_revenue, dtype: float64




```python
print("Expected conditional average profit: %s vs. Average profit: %s" % (
    ggf.conditional_expected_average_profit(
        summary['frequency'],
        summary['monetary_value_average']
    ).mean(),
    summary[summary['frequency']>0]['monetary_value_average'].mean()
))
```

    Expected conditional average profit: 20.41361394230678 vs. Average profit: 18.131239279823276
    

## Calculate the Customer Lifetime Value disconted by DCF and a annual interest rate


```python
# refit the BG model 
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# modelling CLV
summary['clv'] = ggf.customer_lifetime_value(
    bgf,
    summary['frequency'],
    summary['recency'],
    summary['T'],
    summary['monetary_value_average'],
    time=12, # lifetime in months
    discount_rate=0.006 
)
summary['clv'].head(10)
```




    0   52.30111846
    1   20.98403156
    2   57.38228431
    3   47.29754080
    4   21.46389456
    5   52.60841130
    6   14.31632335
    7   27.28834943
    8   14.72406602
    9   15.56113740
    Name: clv, dtype: float64




```python
# describe the distribution
pd.options.display.float_format = '{:.8f}'.format
summary['clv'].describe()

```




    count   10000.00000000
    mean       56.73016752
    std       142.07256046
    min         0.00000003
    25%        13.83895220
    50%        21.00412658
    75%        39.37786598
    max      3623.81967049
    Name: clv, dtype: float64




```python
ax = sns.distplot(summary['clv'], kde=False, rug=True)
ax.set_yscale('log')
plt.suptitle('Distribution of Customer Lifetime Value vs. logarithmic density')

plt.show()
```


    
![png](Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_files/Non-contractual-Customer-Lifetime-Value-with-BG-NBD-model_51_0.png)
    


Although the 75% percentile is at under 40 bucks, few customers with high variance and 4 digit CLV lift the arithmetic mean of CLV way above that number.

It makes sense to segment these different customer types differently to better approach them.
The features from this BG/NBD have been engineered and can be used for such a customer segmentation.

Compare with the original summary df and look at all these new columns:


```python
summary.columns
```




    Index(['customer_id', 'username', 'name', 'sex', 'address', 'mail',
           'birthdate', 'frequency', 'recency', 'T', 'predicted_purchases_14',
           'predicted_purchases_30', 'predicted_purchases_7',
           'monetary_value_average', 'Expected_cond_average_revenue', 'error_rev',
           'clv'],
          dtype='object')



References

[1]	[Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010), “Customer-Base Analysis in a Discrete-Time Noncontractual Setting,” Marketing Science, 29 (6), 1086-1108.](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)

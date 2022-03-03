---
title: Non Fungible Token (NFT) Rarity Analysis with Jaccard Distance
tags: [Python, sample set dissimilarity, Jaccard Distance, Jaccard Similarity, Scraping, NFT, Blockchain]
style: fill
color: primary
comments: true
description: NFT trait set distance calculation with Jaccard Similarity/Distance for rarity measurement of indivual tokens in collections
---


## trait set distance calculation with Jaccard Similarity/Distance

### Overview market macro stats for a market & collection of my choice


```python
# api market macro stats
# for collection officialtimelessapeclub
# on opensea market
import requests, json

url_stats = "https://api.opensea.io/api/v1/collection/officialtimelessapeclub/stats"

r = requests.get(url_stats)

print(f"floor price:", r.json()["stats"]["floor_price"],f"ETH")
print(f"24h Volume:", r.json()["stats"]["one_day_volume"],f"ETH")
print(f"24h sales:", r.json()["stats"]["one_day_sales"])

print(f"Number of sales in last 30 days", r.json()["stats"]["thirty_day_sales"])
print(f"24h AVG price:", r.json()["stats"]["one_day_average_price"],f"ETH")
print(f"Unique owners:", r.json()["stats"]["num_owners"],f"/", r.json()["stats"]["count"] )
```

    floor price: 0.04 ETH
    24h Volume: 0.6194999999999999 ETH
    24h sales: 15.0
    Number of sales in last 30 days 1426.0
    24h AVG price: 0.041299999999999996 ETH
    Unique owners: 2168 / 4645.0
    


```python
import pandas as pd
asset_traits = pd.read_pickle('asset_traits.pkl') # load prepared df with trait variants of tokens 0 to 4094
```


```python
asset_traits.head()
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
      <th>token_id</th>
      <th>MOUTH</th>
      <th>HAT</th>
      <th>EYES</th>
      <th>CLOTH</th>
      <th>BACKGROUND</th>
      <th>SKIN</th>
      <th>WATCH</th>
      <th>LEGENDARY APE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Bubblegum</td>
      <td>Trucker</td>
      <td>3D</td>
      <td>Astronaut</td>
      <td>Cream</td>
      <td>Brown</td>
      <td>Bartier Cantos</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Smirk</td>
      <td>Beanie</td>
      <td>Frown</td>
      <td>Bulletproof Vest</td>
      <td>Cream</td>
      <td>Gray</td>
      <td>Robin Sub</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Bubblegum</td>
      <td>Jungle</td>
      <td>Star</td>
      <td>Army</td>
      <td>Light Green</td>
      <td>Gray</td>
      <td>Parrot Lumos</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Disappointed</td>
      <td>Bunny</td>
      <td>Sad</td>
      <td>Tshirt</td>
      <td>Light Green</td>
      <td>Blue</td>
      <td>Cartoon</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Disappointed</td>
      <td>Boat</td>
      <td>Bored</td>
      <td>Singlet</td>
      <td>Sea Green</td>
      <td>Gold</td>
      <td>Berry Rose</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now we want to compare each token id and its specific trait set to all other tokens and see how different each token really is compared to all other tokens. We apply here the concept of Jaccard Similarity and Distance which are defined as follows:

$$ Jaccard Similarity: J = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| – |A \cup B|} $$


$$ Jaccard Distance: d_J = \frac{|A \cup B| – |A \cap B|}{|A \cup B|} = 1 – J(A, B) $$

We use the Jaccard Distance to measures dissimilarity between its traits and every other token trait set in the same collection. The Jaccard distance facilitates the symmetric difference of the two sets and measures its size, which is put into relation to the union (of these two sets).

I have optimized these numerous iterations of rows with ```df.itertuples()``` already, but still thinking on how to implement the same completely vectorized, if that is possible.

### Update 1 - 03-Mar-22: 

- Label encoding trait variants
- Implementing Multiprocessing Pool (spawn processes for DataFrame chunks)



```python
# Update 1 - 03-Mar-22
# map trait variants to short number codes before number crunching to save 40%+ processing time

dftest = asset_traits.copy()

# all unique trait variants
u =set()
for col in dftest.iloc[:,1:].columns:
    dftest[col] = dftest[col].fillna(0)
    v = set(dftest[col].unique())
    u = u.union(v)

# map unique code across all cols to all trait variants
mapping = {item:i for i, item in enumerate(u)}

for col in dftest.iloc[:,1:].columns:
    dftest[col] = dftest[col].apply(lambda x: mapping[x])
    
dftest.head()
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
      <th>token_id</th>
      <th>MOUTH</th>
      <th>HAT</th>
      <th>EYES</th>
      <th>CLOTH</th>
      <th>BACKGROUND</th>
      <th>SKIN</th>
      <th>WATCH</th>
      <th>LEGENDARY APE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>75</td>
      <td>120</td>
      <td>96</td>
      <td>34</td>
      <td>22</td>
      <td>118</td>
      <td>62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>91</td>
      <td>63</td>
      <td>19</td>
      <td>109</td>
      <td>22</td>
      <td>78</td>
      <td>101</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>75</td>
      <td>84</td>
      <td>64</td>
      <td>3</td>
      <td>114</td>
      <td>78</td>
      <td>90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>89</td>
      <td>42</td>
      <td>33</td>
      <td>102</td>
      <td>114</td>
      <td>106</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>89</td>
      <td>44</td>
      <td>80</td>
      <td>11</td>
      <td>73</td>
      <td>2</td>
      <td>88</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%time
# Update 1 - 03-Mar-22
# multiprocessing pool for the cpu intensive jaccard distances calculations

import multiprocessing
#from multiprocessing import Pool
from itertools import repeat
import pandas as pd
import jaccard # moved jaccard distance calculations code as jaccard function to jaccard.py in order to make multiproc work

collection_slug = 'officialtimelessapeclub'

# Divide asset dataframe into n chunks
n = 6 # define the number of processes (cpu cores)
chunk_size = int(asset_traits.shape[0]/n)+1
chunks = [asset_traits.index[i:i + chunk_size] for i in range(0, asset_traits.shape[0], chunk_size)]

# multiprocess pool - sending chunk indices and collection info as args, df opened directly by jaccard.py
with multiprocessing.Pool() as pool:
    result = pool.starmap(jaccard.jaccard, zip(chunks, repeat(collection_slug)))
    
# Concatenate all chunks back to one
jdist = pd.concat(result)
```

    Wall time: 2min 11s
    

Saving approx. 75% time compared to the single thread code.


```python
%%time
# original single thread solution

import numpy as np
from scipy.spatial.distance import jaccard
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur

df2 = asset_traits.copy()

jdist = pd.DataFrame(columns=['token_id','jd_mean'])
    
# calc jaccard dist for each row
for row1 in df2.itertuples():
    jdlist = []
    # calc against all other rows
    for row2 in df2.itertuples():
        if row1.token_id != row2.token_id:
            jd = jaccard( np.array(row1[2:]), np.array(row2[2:]) ) # jaccard distance
            jdlist.append(jd)     
    jdist.loc[row1.token_id] = [row1.token_id, np.mean(jdlist)] # save the mean of jd of row1
```

    Wall time: 9min 40s
    


```python
# normalize jaccard distances for a score 0 to 1
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
arr = jdist['jd_mean'].to_numpy()
normalized = scaler.fit_transform(arr.reshape(-1, 1))
# write to df
score = pd.DataFrame(columns=['token_id','score'])
score['token_id'] = jdist['token_id']
score['score'] = normalized
```


```python
from scipy import stats
zscores = stats.zscore(arr)
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

plt.figure(figsize=(15,4))
sns.histplot(data=score.score)
plt.title('Normalized scores 0 to 1')
plt.show()
plt.figure(figsize=(15,4))
sns.histplot(data=zscores)
plt.title('z-scores of Jaccard distance means')
plt.show()
```


    
![png](/blog/nft2-blog-update1-multiprocessing_files/nft2-blog-update1-multiprocessing_13_0.png)
    



    
![png](/blog/nft2-blog-update1-multiprocessing_files/nft2-blog-update1-multiprocessing_13_1.png)
    


It becomes obvious of how skewed the distribution of tokens is and how dissimilar the legendary apes on the right really are in this collection, that is 17 standard deviations!


```python
# normalized scores as score
# and the arithmetic mean of the original jaccard dissimilarities (distances)
score['jdist_mean'] = jdist.jd_mean
score.head()
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
      <th>token_id</th>
      <th>score</th>
      <th>jdist_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.085865</td>
      <td>6.452084</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.040798</td>
      <td>6.376137</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.104821</td>
      <td>6.484028</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.068918</td>
      <td>6.423524</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.117876</td>
      <td>6.506029</td>
    </tr>
  </tbody>
</table>
</div>




```python
score.to_pickle('score.pkl')    # save score df to file
```


```python
# best score 1
score.sort_values('score', ascending=False)
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
      <th>token_id</th>
      <th>score</th>
      <th>jdist_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3623</th>
      <td>3623.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
    </tr>
    <tr>
      <th>2533</th>
      <td>2533.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
    </tr>
    <tr>
      <th>106</th>
      <td>106.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
    </tr>
    <tr>
      <th>3888</th>
      <td>3888.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
    </tr>
    <tr>
      <th>1836</th>
      <td>1836.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>916</th>
      <td>916</td>
      <td>0.004328</td>
      <td>6.317152</td>
    </tr>
    <tr>
      <th>332</th>
      <td>332</td>
      <td>0.003526</td>
      <td>6.315804</td>
    </tr>
    <tr>
      <th>3251</th>
      <td>3251</td>
      <td>0.002565</td>
      <td>6.314186</td>
    </tr>
    <tr>
      <th>3068</th>
      <td>3068</td>
      <td>0.000641</td>
      <td>6.310949</td>
    </tr>
    <tr>
      <th>1510</th>
      <td>1510</td>
      <td>0.000000</td>
      <td>6.309871</td>
    </tr>
  </tbody>
</table>
<p>3709 rows × 3 columns</p>
</div>




```python
# let's rank the scores with pandas rank
score['rank'] = score['score'].rank(method='min', ascending=False)
score.sort_values('rank', ascending=True)
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
      <th>token_id</th>
      <th>score</th>
      <th>jdist_mean</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>106</th>
      <td>106.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1836</th>
      <td>1836.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3623</th>
      <td>3623.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2533</th>
      <td>2533.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3888</th>
      <td>3888.0</td>
      <td>1.000000</td>
      <td>7.992449</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>916</th>
      <td>916</td>
      <td>0.004328</td>
      <td>6.317152</td>
      <td>3705.0</td>
    </tr>
    <tr>
      <th>332</th>
      <td>332</td>
      <td>0.003526</td>
      <td>6.315804</td>
      <td>3706.0</td>
    </tr>
    <tr>
      <th>3251</th>
      <td>3251</td>
      <td>0.002565</td>
      <td>6.314186</td>
      <td>3707.0</td>
    </tr>
    <tr>
      <th>3068</th>
      <td>3068</td>
      <td>0.000641</td>
      <td>6.310949</td>
      <td>3708.0</td>
    </tr>
    <tr>
      <th>1510</th>
      <td>1510</td>
      <td>0.000000</td>
      <td>6.309871</td>
      <td>3709.0</td>
    </tr>
  </tbody>
</table>
<p>3709 rows × 4 columns</p>
</div>




```python
#let's check a specific token for its rank
score.loc[['1081']]
# token 1081 is pretty average:
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
      <th>token_id</th>
      <th>score</th>
      <th>jdist_mean</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1081</th>
      <td>1081</td>
      <td>0.076294</td>
      <td>6.438242</td>
      <td>2100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# let us check the current market what rarities are offered at which prices
# for that we have to scrape due to lack of immeadiately available api key 
# which is required for the 'Retrieving assets' api call
# https://docs.opensea.io/reference/getting-assets

import requests
import cloudscraper # can scrape cloudflare sites

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'firefox',
        'platform': 'windows',
        'mobile': False
    }
)
from bs4 import BeautifulSoup as bs
url = "https://opensea.io/collection/officialtimelessapeclub?search[sortAscending]=true&search[sortBy]=PRICE&search[toggles][0]=BUY_NOW"
r = scraper.get(url).text
```


```python
# todo: change to selenium to better deal with opensea uses infinity scrolling

import re
market = pd.DataFrame(columns=['token_id', 'price', 'rank', 'price_rank_product'])

html_code = bs(r, 'html.parser')
cards = html_code.find_all('article', class_=re.compile('^AssetSearchList--asset'))

for c in cards:
    price = c.find('div', class_=re.compile('^Price--amount')).getText(strip=True).split("#",1)[0] if c.find('div', class_=re.compile('^Price--amount')) else "none"
    token = c.find('div', class_=re.compile('^AssetCardFooter--name')).getText(strip=True).split("#",1)[1] if c.find('div', class_=re.compile('^AssetCardFooter--name')) else "none"
    
    token_score = tmp.loc[tmp.token_id == token]['rank'].to_numpy()
    
    # todo: change to selenium to better deal with opensea uses infinity scrolling
    if len(token_score) == 1:
        grade = round(float(price) * token_score[0],1)
    else: #lazy loaded cards
        token_score = str(token_score)
        grade = '-----'
    market.loc[len(market.index)] = [token, price, token_score, grade]
```


```python
market[0:32].sort_values('price')
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
      <th>token_id</th>
      <th>price</th>
      <th>rank</th>
      <th>price_rank_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1328</td>
      <td>0.05</td>
      <td>[3064]</td>
      <td>153.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3073</td>
      <td>0.05</td>
      <td>[3064]</td>
      <td>153.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1220</td>
      <td>0.05</td>
      <td>[1331]</td>
      <td>66.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>209</td>
      <td>0.05</td>
      <td>[3440]</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>210</td>
      <td>0.055</td>
      <td>[3496]</td>
      <td>192.3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>208</td>
      <td>0.059</td>
      <td>[1576]</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1092</td>
      <td>0.059</td>
      <td>[2908]</td>
      <td>171.6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3348</td>
      <td>0.06</td>
      <td>[1745]</td>
      <td>104.7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3290</td>
      <td>0.06</td>
      <td>[2830]</td>
      <td>169.8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>212</td>
      <td>0.064</td>
      <td>[2117]</td>
      <td>135.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3685</td>
      <td>0.065</td>
      <td>[2813]</td>
      <td>182.8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1091</td>
      <td>0.069</td>
      <td>[1864]</td>
      <td>128.6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>966</td>
      <td>0.069</td>
      <td>[1454]</td>
      <td>100.3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1869</td>
      <td>0.069</td>
      <td>[1138]</td>
      <td>78.5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1240</td>
      <td>0.07</td>
      <td>[2873]</td>
      <td>201.1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1430</td>
      <td>0.07</td>
      <td>[3306]</td>
      <td>231.4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>780</td>
      <td>0.07</td>
      <td>[2379]</td>
      <td>166.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>534</td>
      <td>0.07</td>
      <td>[1507]</td>
      <td>105.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3044</td>
      <td>0.07</td>
      <td>[2547]</td>
      <td>178.3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>257</td>
      <td>0.07</td>
      <td>[2971]</td>
      <td>208.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3995</td>
      <td>0.07</td>
      <td>[3647]</td>
      <td>255.3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1080</td>
      <td>0.075</td>
      <td>[2288]</td>
      <td>171.6</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1081</td>
      <td>0.075</td>
      <td>[2100]</td>
      <td>157.5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2811</td>
      <td>0.075</td>
      <td>[2254]</td>
      <td>169.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>677</td>
      <td>0.075</td>
      <td>[2262]</td>
      <td>169.6</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2833</td>
      <td>0.08</td>
      <td>[1996]</td>
      <td>159.7</td>
    </tr>
    <tr>
      <th>25</th>
      <td>166</td>
      <td>0.08</td>
      <td>[1758]</td>
      <td>140.6</td>
    </tr>
    <tr>
      <th>26</th>
      <td>770</td>
      <td>0.08</td>
      <td>[2112]</td>
      <td>169.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1116</td>
      <td>0.08</td>
      <td>[2918]</td>
      <td>233.4</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1455</td>
      <td>0.08</td>
      <td>[824]</td>
      <td>65.9</td>
    </tr>
    <tr>
      <th>29</th>
      <td>49</td>
      <td>0.08</td>
      <td>[1618]</td>
      <td>129.4</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1017</td>
      <td>0.08</td>
      <td>[3393]</td>
      <td>271.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# voila, pick the best deals
market[0:32].sort_values('price_rank_product').head()
# metric still need calibration and weighting as required
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
      <th>token_id</th>
      <th>price</th>
      <th>rank</th>
      <th>price_rank_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>1455</td>
      <td>0.08</td>
      <td>[824]</td>
      <td>65.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1220</td>
      <td>0.05</td>
      <td>[1331]</td>
      <td>66.6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1869</td>
      <td>0.069</td>
      <td>[1138]</td>
      <td>78.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>208</td>
      <td>0.059</td>
      <td>[1576]</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>966</td>
      <td>0.069</td>
      <td>[1454]</td>
      <td>100.3</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

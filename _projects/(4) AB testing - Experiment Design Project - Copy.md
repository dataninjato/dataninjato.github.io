---
name: Udacity AB testing Online Experiment Design and Analysis - Final Project in Python
tools: [AB testing, Udacity, Hypothesis, Experiment Design, Python]
image: '/blog/Final Project_ Experiment Screenshot.png'
description: AB testing Experiment Design Project
external_url: 
---


# Udacity AB testing Online Experiment Design and Analysis - Final Project in Python

## Experiment Design Project

Experiment Overview: Free Trial Screener

At the time of this experiment, Udacity courses currently have two options on the course overview page: "start free trial", and "access course materials". If the student clicks "start free trial", they will be asked to enter their credit card information, and then they will be enrolled in a free trial for the paid version of the course. After 14 days, they will automatically be charged unless they cancel first. If the student clicks "access course materials", they will be able to view the videos and take the quizzes for free, but they will not receive coaching support or a verified certificate, and they will not submit their final project for feedback.


In the experiment, Udacity tested a change where if the student clicked "start free trial", they were asked how much time they had available to devote to the course. If the student indicated 5 or more hours per week, they would be taken through the checkout process as usual. If they indicated fewer than 5 hours per week, a message would appear indicating that Udacity courses usually require a greater time commitment for successful completion, and suggesting that the student might like to access the course materials for free. At this point, the student would have the option to continue enrolling in the free trial, or access the course materials for free instead. This screenshot shows what the experiment looks like.


The hypothesis was that this might set clearer expectations for students upfront, thus reducing the number of frustrated students who left the free trial because they didn't have enough time—without significantly reducing the number of students to continue past the free trial and eventually complete the course. If this hypothesis held true, Udacity could improve the overall student experience and improve coaches' capacity to support students who are likely to complete the course.


The unit of diversion is a cookie, although if the student enrolls in the free trial, they are tracked by user-id from that point forward. The same user-id cannot enroll in the free trial twice. For users that do not enroll, their user-id is not tracked in the experiment, even if they were signed in when they visited the course overview page.

Please refer to the complete <a href='https://docs.google.com/document/d/16OX2KDSHI9mSCriyGIATpRGscIW2JmByMd0ITqKYvNg/edit#' >Final Project Instructions</a>

To highlight the Hypothesis:

A popup screener might set clearer expectations for students upfront enrolling, thus reducing the number of frustrated students who leave the free trial because they didn't have enough time, at the same time without significantly reducing the number of students to continue past the free trial and eventually completing the course

### Phases of interest


```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Diversion="cookie diversion", Start='2021-01-01', End='2021-01-02', Phase="unique user cookie course overview page"),
    dict(Diversion="cookie diversion", Start='2021-01-04', End='2021-01-05', Phase="unique user cookie course overview page"),
    dict(Diversion="cookie diversion", Start='2021-01-05', End='2021-01-06', Phase="unique user cookie course overview page"),
    dict(Diversion="cookie diversion", Start='2021-01-06', End='2021-01-07', Phase="unique user cookie course overview page"),
    dict(Diversion="both diversions", Start='2021-01-06', End='2021-01-07', Phase="'Start free trial' button click"),
    dict(Diversion="user-id diversion", Start='2021-01-05', End='2021-02-20', Phase="user-id login"),
    dict(Diversion="user-id diversion", Start='2021-01-07', End='2021-01-21', Phase="checkout & free trial"),
    dict(Diversion="user-id diversion", Start='2021-01-21', End='2021-02-20', Phase="enrolled past 14 days"),
    dict(Diversion="user-id diversion", Start='2021-02-20', End='2021-02-28', Phase="unenrolled"),
])

# fig = px.timeline(df, x_start="Start", x_end="End", y="Phase", color="Diversion",
#                   color_discrete_sequence=px.colors.qualitative.Vivid,
#                  title='Exemplary phases of the Udacity user signup process' )
# fig.show()
```


```python
df.head()
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
      <th>Diversion</th>
      <th>Start</th>
      <th>End</th>
      <th>Phase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cookie diversion</td>
      <td>2021-01-01</td>
      <td>2021-01-02</td>
      <td>unique user cookie course overview page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cookie diversion</td>
      <td>2021-01-04</td>
      <td>2021-01-05</td>
      <td>unique user cookie course overview page</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cookie diversion</td>
      <td>2021-01-05</td>
      <td>2021-01-06</td>
      <td>unique user cookie course overview page</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cookie diversion</td>
      <td>2021-01-06</td>
      <td>2021-01-07</td>
      <td>unique user cookie course overview page</td>
    </tr>
    <tr>
      <th>4</th>
      <td>both diversions</td>
      <td>2021-01-06</td>
      <td>2021-01-07</td>
      <td>'Start free trial' button click</td>
    </tr>
  </tbody>
</table>
</div>



**Metric Choice**

List which metrics you will use as invariant metrics and evaluation
metrics here. (These should be the same metrics you chose in the
\"Choosing Invariant Metrics\" and \"Choosing Evaluation Metrics\"
quizzes.)

 **Metrics on the table**|**Suitability check**|**Choice**
:-----:|:-----:|:-----:
Number of cookies: That is, number of unique cookies to view the course overview page. (dmin=3000)|This is clearly an invariant metric as its number is not subject to change and should be the similar before and after the launched change. Therefore, it is not useable as evaluation metric.|Invariant
Number of user-ids: That is, number of users who enroll in the free trial. (dmin=50)|The count of user-ids is subject of evaluation of this experiment and is supposed to change (not invariant). The use of a simple count may make it difficult to compare to pre-period with overall different number of users.|-
Number of clicks: That is, number of unique cookies to click the "Start free trial" button (which happens before the free trial screener is trigger). (dmin=240)|The number of clicks to start free trial is unlikely to change as it remains unchanged by the experiment, so invariant in this design. The use of a simple count number may make it difficult to compare to pre-period.|Invariant
Click-through-probability: That is, number of unique cookies to click the "Start free trial" button divided by number of unique cookies to view the course overview page. (dmin=0.01)|The CTP is a much better candidate as the previous, invariant by nature and comparable.|Invariant
Gross conversion: That is, number of user-ids to complete checkout and enroll in the free trial divided by number of unique cookies to click the "Start free trial" button. (dmin= 0.01)|This ratio is a good complementary evaluation metric. It may be a sensitive metric when others like net conversion do not show a change, in case the experiment fails to retain users better.|Evaluation
Retention: That is, number of user-ids to remain enrolled past the 14-day boundary (and thus make at least one payment) divided by number of user-ids to complete checkout. (dmin=0.01)|Retention is defined as fraction of both net conversion over gross conversion. As a derived metric of these two conversions it is un clear if it can add value to an evaluation and what direction to expect really. though we will not use it as evaluation metric I continue to calculate and look at it.|-
Net conversion: That is, number of user-ids to remain enrolled past the 14-day boundary (and thus make at least one payment) divided by the number of unique cookies to click the "Start free trial" button. (dmin= 0.0075)|This ratio is a subsegment of gross conversion only regarding user-ids past 14-day boundary. This ratio is expected to increase significantly, although the absolute number of user-ids might be lower.|Evaluation

**Measuring Standard Deviation**

List the standard deviation of each of your evaluation metrics. (These
should be the answers from the \"Calculating standard deviation\" quiz.)

*For each of your evaluation metrics, indicate whether you think the
analytic estimate would be comparable to the the empirical variability,
or whether you expect them to be different (in which case it might be
worth doing an empirical estimate if there is time). Briefly give your
reasoning in each case.*


```python
# Baseline Values

# udacity baseline google sheet 
google_sheet_url = 'https://docs.google.com/spreadsheets/d/1MYNUtC47Pg8hdoCjOXaHqF-thheGpUshrFA21BAJnNc/export?format=csv&gid=0'

import pandas as pd
df=pd.read_csv(google_sheet_url, names=['observation', 'baseline'], encoding="utf-8")

```


```python
#df.iloc[2,0] = 'Enrollments'
df['shortname'] = ['cookies',
              'clicks',
              'enrollments',
              'p_click_through',
              'p_enroll_given_click',
               'p_payment_given_enroll',
               'p_payment_given_click'
              ]
df['metric'] = ['',
              '',
              '',
              '',
              'gross_conversion',
               'retention',
               'net_conversion',
              ]
df['dmin'] = [3000,240,50,0.01,0.01,0.01,0.0075]
df['potent_metric'] = [False,False,False,False,True,True,True]
df = df.set_index('shortname')
#df['n'] = 0
```


```python
df
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
      <th>observation</th>
      <th>baseline</th>
      <th>metric</th>
      <th>dmin</th>
      <th>potent_metric</th>
    </tr>
    <tr>
      <th>shortname</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cookies</th>
      <td>Unique cookies to view course overview page pe...</td>
      <td>40000.000000</td>
      <td></td>
      <td>3000.0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>clicks</th>
      <td>Unique cookies to click "Start free trial" per...</td>
      <td>3200.000000</td>
      <td></td>
      <td>240.0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>enrollments</th>
      <td>Enrollments per day:</td>
      <td>660.000000</td>
      <td></td>
      <td>50.0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>p_click_through</th>
      <td>Click-through-probability on "Start free trial":</td>
      <td>0.080000</td>
      <td></td>
      <td>0.0100</td>
      <td>False</td>
    </tr>
    <tr>
      <th>p_enroll_given_click</th>
      <td>Probability of enrolling, given click:</td>
      <td>0.206250</td>
      <td>gross_conversion</td>
      <td>0.0100</td>
      <td>True</td>
    </tr>
    <tr>
      <th>p_payment_given_enroll</th>
      <td>Probability of payment, given enroll:</td>
      <td>0.530000</td>
      <td>retention</td>
      <td>0.0100</td>
      <td>True</td>
    </tr>
    <tr>
      <th>p_payment_given_click</th>
      <td>Probability of payment, given click</td>
      <td>0.109313</td>
      <td>net_conversion</td>
      <td>0.0075</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Standard Deviation based on given probabilities

Standard deviation of binomial distributions where successes are used in the form of fraction / proportion of successes (as in our given baseline values).
$$ SE = \sqrt{\frac{p*(1-p)}{n}} $$

sample based on n=5000 pageviews/cookies with p=probability of success



```python
df['reference_value'] = ['cookie', 'click', 'enroll', 'cookie', 'click', 'enroll', 'click']
df.loc[df['reference_value'] == 'cookie', 'n'] = 5000
df.loc[df['reference_value'] == 'click', 'n'] = df.loc['cookies','n'] * df.loc['p_click_through','baseline'] 
df.loc[df['reference_value'] == 'enroll', 'n'] = df.loc['clicks','n'] * df.loc['p_enroll_given_click','baseline']
```


```python
import numpy as np

def StandardError(p,n):
    stdev = np.sqrt(p*(1-p)/n)
    return stdev

df['SE'] = df[df['potent_metric']==True].apply( lambda row: StandardError(row['baseline'], row['n']), axis=1)

df
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
      <th>observation</th>
      <th>baseline</th>
      <th>metric</th>
      <th>dmin</th>
      <th>potent_metric</th>
      <th>reference_value</th>
      <th>n</th>
      <th>SE</th>
    </tr>
    <tr>
      <th>shortname</th>
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
      <th>cookies</th>
      <td>Unique cookies to view course overview page pe...</td>
      <td>40000.000000</td>
      <td></td>
      <td>3000.0000</td>
      <td>False</td>
      <td>cookie</td>
      <td>5000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>clicks</th>
      <td>Unique cookies to click "Start free trial" per...</td>
      <td>3200.000000</td>
      <td></td>
      <td>240.0000</td>
      <td>False</td>
      <td>click</td>
      <td>400.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>enrollments</th>
      <td>Enrollments per day:</td>
      <td>660.000000</td>
      <td></td>
      <td>50.0000</td>
      <td>False</td>
      <td>enroll</td>
      <td>82.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>p_click_through</th>
      <td>Click-through-probability on "Start free trial":</td>
      <td>0.080000</td>
      <td></td>
      <td>0.0100</td>
      <td>False</td>
      <td>cookie</td>
      <td>5000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>p_enroll_given_click</th>
      <td>Probability of enrolling, given click:</td>
      <td>0.206250</td>
      <td>gross_conversion</td>
      <td>0.0100</td>
      <td>True</td>
      <td>click</td>
      <td>400.0</td>
      <td>0.020231</td>
    </tr>
    <tr>
      <th>p_payment_given_enroll</th>
      <td>Probability of payment, given enroll:</td>
      <td>0.530000</td>
      <td>retention</td>
      <td>0.0100</td>
      <td>True</td>
      <td>enroll</td>
      <td>82.5</td>
      <td>0.054949</td>
    </tr>
    <tr>
      <th>p_payment_given_click</th>
      <td>Probability of payment, given click</td>
      <td>0.109313</td>
      <td>net_conversion</td>
      <td>0.0075</td>
      <td>True</td>
      <td>click</td>
      <td>400.0</td>
      <td>0.015602</td>
    </tr>
  </tbody>
</table>
</div>



The standard errors are calculated analytically as the unit of diversion for each evaluation metric is identical to the unit of analysis (denominating factor). Therefore empirical estimation is not required.

The sample proportions are approximated by normal distributions as the size of all referenced sample sizes are deemed as sufficiently large, which can be formally confirmed by the 3-Standard-Deviation Rule for Normal Approximations of Binomial distributions. As both probabilities are not at extremes and the sample sizes n are large enough this check is waived here.

### Sizing the experiment

**Number of Samples vs. Power**



*Indicate whether you will use the Bonferroni correction during your
analysis phase, and give the number of pageviews you will need to power
you experiment appropriately. (These should be the answers from the
\"Calculating Number of Pageviews\" quiz.)*

calculate minimum sample sizes in order to have sufficient statistical power, so having enough numbers to be able to tell if there is a signifcant difference between control and treatment group. 

$ Gross conversion = C $

$ alpha = 0.05 \quad(Type I error) $

$ beta = 0.2 \quad(Type II error)$

$ H_{0}: C_{cont} - C_{treat} = 0 $

$ H_{A}: C_{cont} - C_{treat} \neq 0 $


```python
from scipy.stats import norm

#Inputs: required alpha value
#Returns: z-score for given alpha
def z_score(alpha):
    return norm.ppf(alpha)

# Inputs p-baseline conversion rate which is our estimated p and dmin detectable change
# Returns
def sds(p,d):
    sd1=np.sqrt(2*p*(1-p))
    sd2=np.sqrt(p*(1-p)+(p+d)*(1-(p+d)))
    sds=[sd1,sd2]
    return sds

# In: sd1-sd for the baseline, 
#     sd2-sd for the expected change,
#     alpha,
#     beta,
#     dmin,
#     p-baseline estimate p
# Returns: the minimum sample size required per group according to metric denominator
def sampSize(sds,alpha,beta,d):
    size=(z_score(1-alpha/2)*sds[0]+z_score(1-beta)*sds[1])**2/d**2
    return size

# bonferroni correction is needed to control the family-wise error rate as both metrics are not independent, 
# and both metrics share the Clicks number therefore the chance for a false positive is certainly higher. 
# it is assumed that one significant test of a single metric is sufficient to reject the Null Hypothesis
alpha = 0.05
numtests = 2 # retention metric excluded
bonferroni_individual_alpha = alpha / numtests

beta = 0.2

df['samplesize']=sampSize(sds(df.baseline[3:7],df.dmin[3:7]),bonferroni_individual_alpha, beta, df.dmin[3:7]) # only probab.
df
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
      <th>observation</th>
      <th>baseline</th>
      <th>metric</th>
      <th>dmin</th>
      <th>potent_metric</th>
      <th>reference_value</th>
      <th>n</th>
      <th>SE</th>
      <th>samplesize</th>
    </tr>
    <tr>
      <th>shortname</th>
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
      <th>cookies</th>
      <td>Unique cookies to view course overview page pe...</td>
      <td>40000.000000</td>
      <td></td>
      <td>3000.0000</td>
      <td>False</td>
      <td>cookie</td>
      <td>5000.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>clicks</th>
      <td>Unique cookies to click "Start free trial" per...</td>
      <td>3200.000000</td>
      <td></td>
      <td>240.0000</td>
      <td>False</td>
      <td>click</td>
      <td>400.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>enrollments</th>
      <td>Enrollments per day:</td>
      <td>660.000000</td>
      <td></td>
      <td>50.0000</td>
      <td>False</td>
      <td>enroll</td>
      <td>82.5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>p_click_through</th>
      <td>Click-through-probability on "Start free trial":</td>
      <td>0.080000</td>
      <td></td>
      <td>0.0100</td>
      <td>False</td>
      <td>cookie</td>
      <td>5000.0</td>
      <td>NaN</td>
      <td>14204.630379</td>
    </tr>
    <tr>
      <th>p_enroll_given_click</th>
      <td>Probability of enrolling, given click:</td>
      <td>0.206250</td>
      <td>gross_conversion</td>
      <td>0.0100</td>
      <td>True</td>
      <td>click</td>
      <td>400.0</td>
      <td>0.020231</td>
      <td>31270.939498</td>
    </tr>
    <tr>
      <th>p_payment_given_enroll</th>
      <td>Probability of payment, given enroll:</td>
      <td>0.530000</td>
      <td>retention</td>
      <td>0.0100</td>
      <td>True</td>
      <td>enroll</td>
      <td>82.5</td>
      <td>0.054949</td>
      <td>47335.925254</td>
    </tr>
    <tr>
      <th>p_payment_given_click</th>
      <td>Probability of payment, given click</td>
      <td>0.109313</td>
      <td>net_conversion</td>
      <td>0.0075</td>
      <td>True</td>
      <td>click</td>
      <td>400.0</td>
      <td>0.015602</td>
      <td>33170.892254</td>
    </tr>
  </tbody>
</table>
</div>



Let's multiply the with each reference value denominated sample sizes into the total number of minimum pageviews required for each metric.



```python
df.loc[df.metric == 'gross_conversion', 'required_pageviews'] = df.loc[df.metric == 'gross_conversion', 'samplesize'] / df.loc['p_click_through', 'baseline'] * 2 # 2 groups A + B
df.loc[df.metric == 'gross_conversion', 'required_pageviews'] = df.loc[df.metric == 'gross_conversion', 'required_pageviews'].astype(int)

```


```python
df.loc[df.metric == 'retention', 'required_pageviews'] = df.loc[df.metric == 'retention', 'samplesize'] / df.loc['p_click_through', 'baseline'] / df.loc[df.metric == 'gross_conversion', 'baseline'][0] * 2 # 2 groups A + B
df.loc[df.metric == 'retention', 'required_pageviews'] = df.loc[df.metric == 'retention', 'required_pageviews'].astype(int)
```


```python
df.loc[df.metric == 'net_conversion', 'required_pageviews'] = df.loc[df.metric == 'net_conversion', 'samplesize'] / df.loc['p_click_through', 'baseline'] * 2 # 2 groups A + B
df.loc[df.metric == 'net_conversion', 'required_pageviews'] = df.loc[df.metric == 'net_conversion', 'required_pageviews'].astype(int)

```

### **Duration vs. Exposure**


*Indicate what fraction of traffic you would divert to this experiment
and, given this, how many days you would need to run the experiment.
(These should be the answers from the \"Choosing Duration and Exposure\"
quiz.)*

Give your reasoning for the fraction you chose to divert. How risky do
you think this experiment would be for Udacity?

from the previously calculated number of required pageviews we can estimate the approximate the duration of days the experiment has to run assuming given constant website traffic


```python
df['duration'] = df.apply( lambda row: row.required_pageviews / df.loc['cookies', 'baseline'], axis=1)
```


```python
determine_duration = df[df.potent_metric==True & (df.metric!='retention')][['metric', 'required_pageviews', 'duration']]
determine_duration
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
      <th>metric</th>
      <th>required_pageviews</th>
      <th>duration</th>
    </tr>
    <tr>
      <th>shortname</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p_enroll_given_click</th>
      <td>gross_conversion</td>
      <td>781773.0</td>
      <td>19.544325</td>
    </tr>
    <tr>
      <th>p_payment_given_click</th>
      <td>net_conversion</td>
      <td>829272.0</td>
      <td>20.731800</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'the test has to run at least {int(np.ceil(max(determine_duration.duration)))} days assuming constant website traffic')
```

    the test has to run at least 21 days assuming constant website traffic
    

also to be taken into account is the 14 days trial period that delays enrollment and payment data, therefore these 14 days have to be added, amounting to **35 days** in total

This tested change with the screener trial popup is not considered high or elevated risk, therefore all traffic can be diverted to the test. As the test has to run for 5 weeks already there is not much wiggle room to prolong that anyway by diverting only partial traffic, then other metrics would have to be considered.

## Experiment Analysis


```python
# https://stackoverflow.com/questions/19611729/
def load_from_gspreadsheet(sheet_name, key):
    url = 'https://docs.google.com/spreadsheets/d/{key}/gviz/tq?tqx=out:csv&sheet={sheet_name}&headers=1'.format(
        key=key, sheet_name=sheet_name.replace(' ', '%20'))

    #log.info('Loading google spreadsheet from {}'.format(url))

    df = pd.read_csv(url)
    return df.drop([col for col in df.columns if col.startswith('Unnamed')], axis=1)
```


```python
cont = load_from_gspreadsheet('Control', '1Mu5u9GrybDdska-ljPXyBjTpdZIUev_6i7t4LRDfXM8')
cont.head()
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
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7723</td>
      <td>687</td>
      <td>134.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon, Oct 13</td>
      <td>10511</td>
      <td>909</td>
      <td>167.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue, Oct 14</td>
      <td>9871</td>
      <td>836</td>
      <td>156.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wed, Oct 15</td>
      <td>10014</td>
      <td>837</td>
      <td>163.0</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
exp = load_from_gspreadsheet('Experiment', gsheet_results_key)
exp.head()
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
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7716</td>
      <td>686</td>
      <td>105.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9288</td>
      <td>785</td>
      <td>116.0</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon, Oct 13</td>
      <td>10480</td>
      <td>884</td>
      <td>145.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue, Oct 14</td>
      <td>9867</td>
      <td>827</td>
      <td>138.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wed, Oct 15</td>
      <td>9793</td>
      <td>832</td>
      <td>140.0</td>
      <td>94.0</td>
    </tr>
  </tbody>
</table>
</div>



### Sanity Checks

For each of your invariant metrics, give the 95% confidence interval for the value you expect to observe, the actual observed value, and whether the metric passes your sanity check. (These should be the answers from the "Sanity Checks" quiz.)


#### Sanity check of First invariant metric: Pageviews/cookies


```python
# proportion checking

# H0: p = 0.5 # equal click proportions 
```


```python
cont.Pageviews.sum()
```




    345543




```python
exp.Pageviews.sum()
```




    344660




```python
p = 0.5

stdev = np.sqrt(p*(1-p)/(cont.Pageviews.sum() + exp.Pageviews.sum()))
stdev
```




    0.0006018407402943247




```python
# zscore
alpha = 0.05
z= 1.96

# margin of error
margin_of_error = stdev * z
margin_of_error 
```




    0.0011796078509768765




```python
# confidence interval construction

min_ci = p - margin_of_error
max_ci = p + margin_of_error

print('confidence interval:')
print(round(min_ci,4))
print(p)
print(round(max_ci,4))
```

    confidence interval:
    0.4988
    0.5
    0.5012
    


```python
cont_cookie_prop = round(cont.Pageviews.sum() / (cont.Pageviews.sum() + exp.Pageviews.sum()),4)
if(cont_cookie_prop > min_ci and cont_cookie_prop < max_ci):
    print(f'sanity check passed: {cont_cookie_prop} is inside the confidence interval')
```

    sanity check passed: 0.5006 is inside the confidence interval
    

#### Sanity check of second Invariant metric: Clicks


```python
# proportion checking

# H0: p = 0.5 # equal click proportions 
```


```python
p = 0.5

stdev = np.sqrt(p*(1-p)/(cont.Clicks.sum() + exp.Clicks.sum()))
stdev
```




    0.002099747079699252




```python
# zscore
alpha = 0.05
z= 1.96

# margin of error
margin_of_error = stdev * z

# confidence interval construction

min_ci = p - margin_of_error
max_ci = p + margin_of_error

print('confidence interval:')
print(round(min_ci,4))
print(p)
print(round(max_ci,4))

```

    confidence interval:
    0.4959
    0.5
    0.5041
    


```python
cont_click_prop = cont.Clicks.sum() / (cont.Clicks.sum() + exp.Clicks.sum())
if(cont_click_prop > min_ci and cont_click_prop < max_ci):
    print('sanity check passed - '+str(cont_click_prop)+' is inside the confidence interval')

```

    sanity check passed - 0.5004673474066628 is inside the confidence interval
    

#### Sanity check of third Invariant metric: CTP


```python
# checking differences in probability-type-numbers

# H0: d = 0
```


```python
# work with Pandas Series
ctp_cont = cont.Clicks.sum() / cont.Pageviews.sum()
ctp_exp = exp.Clicks.sum() / exp.Pageviews.sum()

d_hat = ctp_exp - ctp_cont 
d_hat
```




    5.662709158693602e-05




```python
# gross conversion across both groups
p_pooled_hat = (cont.Clicks.sum() + exp.Clicks.sum()) / (cont.Pageviews.sum() + exp.Pageviews.sum())
round(p_pooled_hat,4)
```




    0.0822




```python
# sample standard deviations

# SE_pooled
SE_pool = np.sqrt(p_pooled_hat*(1-p_pooled_hat) * (1/cont.Pageviews.sum() + 1/exp.Pageviews.sum()) )
SE_pool
```




    0.0006610608156387222




```python
import scipy.stats as st

# default one tail
# two tails needed
p_value = 0.975  # (1-.05/2)
z = st.norm.ppf(p_value)
z
```




    1.959963984540054




```python
# margin of error
margin_of_error = SE_pool * z
margin_of_error 
```




    0.001295655390242568




```python
# confidence interval construction
# The expected difference d between the two groups is zero. 


min_ci = 0 - margin_of_error
max_ci = 0 + margin_of_error

print('confidence interval:')
print(round(min_ci,4))
print(0)
print(round(max_ci,4))
```

    confidence interval:
    -0.0013
    0
    0.0013
    


```python
if(d_hat > min_ci and d_hat < max_ci):
    print(f'sanity check passed: d_hat {d_hat} is inside the confidence interval')

```

    sanity check passed: d_hat 5.662709158693602e-05 is inside the confidence interval
    

All sanity checks have passed successfully - we move on to measure the effect size of the experiment.

## Result Analysis

### Practical and Statistical Significance - Effect Size Tests

*For each of your evaluation metrics, give a 95% confidence interval
around the difference between the experiment and control groups.
Indicate whether each metric is statistically and practically
significant. (These should be the answers from the \"Effect Size Tests\"
quiz.)*

#### Metric: Gross conversion


```python
# disregard cookies which have not completed 14 days trial phase
cont_non_trial = cont.loc[~cont.Enrollments.isnull()]
exp_non_trial = exp.loc[~exp.Enrollments.isnull()]

print(len(exp))
print(len(exp_non_trial))

# gross conversions per group
cont_gc = cont_non_trial.Enrollments.sum() / cont_non_trial.Clicks.sum()
exp_gc = exp_non_trial.Enrollments.sum() / exp_non_trial.Clicks.sum()

# gross conversion across both groups
pooled_gc_hat = (cont_non_trial.Enrollments.sum() + exp_non_trial.Enrollments.sum()) / (cont_non_trial.Clicks.sum() + exp_non_trial.Clicks.sum())
pooled_gc_hat
```

    37
    23
    




    0.20860706740369866




```python
# SE_pooled
SE_gc_pool = np.sqrt(pooled_gc_hat*(1-pooled_gc_hat) * (1/cont_non_trial.Clicks.sum() + 1/exp_non_trial.Clicks.sum()) )
SE_gc_pool
```




    0.004371675385225936



Null Hypothesis
$$ H_0: d = 0 \qquad(\hat{p}_{cont} = \hat{p}_{exp})\qquad  \hat{d} \sim N(0, SE_{pool})  $$


```python

d_hat = exp_gc - cont_gc
d_hat
```




    -0.020554874580361565




```python
# bonferroni correction
alpha = 0.05
numtests = 2 # retention metric excluded
bonferroni_individual_alpha = alpha / numtests

# margin of error
p_value = 1 - (bonferroni_individual_alpha / 2) # two tails
#print(p_value)

z = st.norm.ppf(p_value)
#print(z)
m = SE_gc_pool * z
m
```




    0.00979868513264882




```python
# confidence interval construction

min_ci = d_hat - m
max_ci = d_hat + m
print('confidence interval:')
print(round(min_ci,4))
print(round(d_hat,4))
print(round(max_ci,4))
```

    confidence interval:
    -0.0304
    -0.0206
    -0.0108
    


```python
if(0 >= min_ci and 0 <= max_ci):
    print('Null hypothesis cannot be rejected as zero is inside the confidence interval')
else:
    print('Null hypothesis can be rejected as zero is outside the confidence interval')
    print('as per experiment design there is a significant reason to doubt the Null hypothesis and') 
    print('conclude that there is indeed a change in the data between control and treatment groups')

```

    Null hypothesis can be rejected as zero is outside the confidence interval
    as per experiment design there is a significant reason to doubt the Null hypothesis and
    conclude that there is indeed a change in the data between control and treatment groups
    

#### Practical Significance Check


```python
# Gross Conversion
dmin_gc = df.loc[df['metric'] == 'gross_conversion', 'dmin'][0]

if (abs(d_hat) > dmin_gc):
    print(f'This observed change {round(abs(d_hat),4)} is practically signficant, as larger than {dmin_gc} and considered worth the implementation')
else:
    print(f'This observed change {round(abs(d_hat),4)} is not practically signficant, as smaller than {dmin_gc} and not considered worth the implementation')
```

    This observed change 0.0206 is practically signficant, as larger than 0.01 and considered worth the implementation
    

#### Metric: Net conversion


```python
# disregard cookies which have not completed 14 days trial phase
#cont_non_trial = cont.loc[~cont.Enrollments.isnull()]
#exp_non_trial = exp.loc[~exp.Enrollments.isnull()]

# net conversions per group
cont_nc = cont_non_trial.Payments.sum() / cont_non_trial.Clicks.sum()
exp_nc = exp_non_trial.Payments.sum() / exp_non_trial.Clicks.sum()

# gross conversion across both groups
pooled_nc_hat = (cont_non_trial.Payments.sum() + exp_non_trial.Payments.sum()) / (cont_non_trial.Clicks.sum() + exp_non_trial.Clicks.sum())
pooled_nc_hat
```




    0.1151274853124186




```python
# sample standard deviations
#stdev_cont = (cont_gc*(1-cont_gc))**0.5
#stdev_exp = (exp_gc*(1-exp_gc))**0.5

# SE_pooled
nc_SE_pool = np.sqrt(pooled_nc_hat*(1-pooled_nc_hat) * (1/cont_non_trial.Clicks.sum() + 1/exp_non_trial.Clicks.sum()) )
nc_SE_pool
```




    0.0034341335129324238



$$ H_0: d = 0 \qquad(\hat{p}_{cont} = \hat{p}_{exp})\qquad  \hat{d} \sim N(0, SE_{pool})  $$


```python

nc_d_hat = exp_nc - cont_nc
nc_d_hat
```




    -0.0048737226745441675




```python
# bonferroni correction
alpha = 0.05
numtests = 2 # retention metric excluded
bonferroni_individual_alpha = alpha / numtests

# margin of error
p_value = 1 - (bonferroni_individual_alpha / 2) # two tails
#p_value = 0.975
print(p_value)
z = st.norm.ppf(p_value)
print(z)
m = nc_SE_pool * z
print(m)
```

    0.9875
    2.241402727604947
    0.007697276222846293
    


```python
# confidence interval construction

nc_min_ci = nc_d_hat - m
nc_max_ci = nc_d_hat + m
print(round(nc_min_ci,4))
print(round(nc_d_hat, 4))
print(round(nc_max_ci,4)) 
```

    -0.0126
    -0.0049
    0.0028
    


```python
if(0 >= nc_min_ci and 0 <= nc_max_ci):
    print('The Null hypothesis cannot be rejected as zero is inside the confidence interval,')
    print('as per experiment design there is no significant reason to doubt the Null hypothesis and conclude that there is no difference')
else:
    print('Null hypothesis can be rejected as zero is outside the confidence interval')
```

    The Null hypothesis cannot be rejected as zero is inside the confidence interval,
    as per experiment design there is no significant reason to doubt the Null hypothesis and conclude that there is no difference
    

This practical signficance test is waived for the metric net conversion as it was not statistical signficant.

### non parametric sign test

For each of your evaluation metrics, do a sign test using the day-by-day data, and report the p-value of the sign test and whether the result is statistically significant. (These should be the answers from the "Sign Tests" quiz.)



```python
# prepare diff series

cont_gc_daily = cont_non_trial.Enrollments / cont_non_trial.Clicks
exp_gc_daily = exp_non_trial.Enrollments / exp_non_trial.Clicks
diff_hat_gc_daily = exp_gc_daily - cont_gc_daily

cont_nc_daily = cont_non_trial.Payments / cont_non_trial.Clicks
exp_nc_daily = exp_non_trial.Payments / exp_non_trial.Clicks
diff_hat_nc_daily = exp_nc_daily - cont_nc_daily
```


```python
# first approach: built-in statsmodels function sign_test

from statsmodels.sandbox.descstats import sign_test

# gross conversion
print(f'gc sign test p-value: {sign_test(diff_hat_gc_daily, 0)[1]}')

# net conversion
print(f'nc sign test p-value: {sign_test(diff_hat_nc_daily, 0)[1]}')
```

    gc sign test p-value: 0.002599477767944336
    nc sign test p-value: 0.6776394844055176
    


```python
# second approach: calculate manually with which we are able to state on how many days there is a significant change
#                  instead of just receiving the result itself as in first approach

def signtest_diffseries(diff_series, alternative = "two_sided"):
    from scipy.stats import binom
    pos_diff_count = len([k for k in diff_series if k > 0])
    timeunits = len(diff_series)
    median_count = np.median(np.arange(0, timeunits + 1))
    
    if (alternative == "two_sided") and (pos_diff_count >= median_count):
      p_value = round(2 * (1 - binom.cdf((pos_diff_count - 1), timeunits, 0.5)),4)
    elif pos_diff_count < median_count:
      p_value = round(2 * binom.cdf(pos_diff_count, timeunits, 0.5),4)
    
    print(f"Number of positive time units: {pos_diff_count} out of a sample size of {timeunits} time units")
    if(p_value < 0.05):
        print(f"The p-value of {p_value} does indicate a significant change in the data. The Null hypothesis H0: d=0 has to be rejected")
    else:
        print(f"The p-value of {p_value} does not indicate a significant change in the data.")
```


```python
signtest_diffseries(diff_hat_gc_daily)
```

    Number of positive time units: 4 out of a sample size of 23 time units
    The p-value of 0.0026 does indicate a significant change in the data. The Null hypothesis H0: d=0 has to be rejected
    


```python
signtest_diffseries(diff_hat_nc_daily)
```

    Number of positive time units: 10 out of a sample size of 23 time units
    The p-value of 0.6776 does not indicate a significant change in the data.
    

Technically both sign test implementations confirm the change in the gross conversion to be significant, while there is no change detected for the net conversion.

**Summary**

The Bonferroni correction was used to control the family wise error rate, as both eval metrics are similar und share the same denominator. The sign test have been conducted to double check validity of the findings, both sign tests have confirmed that Gross conversion changed signficantly while Net conversion did not.

**Recommendation**

To see a change in the gross conversion and simultaneously no change in the net conversion is unexpected and casts doubt on the effectiveness of the implemented change. The implementation is therefore not recommended for the time being in favor of a new approach and different views on the problem on this very topic, which is outlined in a follow-up experiment setup.

## Follow-Up Experiment

Give a high-level description of the follow up experiment you would run,
what your hypothesis would be, what metrics you would want to measure,
what your unit of diversion would be, and your reasoning for these
choices

The change of the current experiment was a query modal window **before** the trial was clicked and that was obviously too timid of a change to be really significant.
The alternative could be to track free trial users (user-ids as unit of diversion) in their learning progress and see if their time devoted to the course is inline with the upfront required 5 hours per week or not. For students trailing behind this could be mirrored to them by text or charts making them aware and consider leaving the trial and instead audit the course resources for free at their own pace, options to re-join the trial anytime available thereafter. This could free companies coaching resourses more effectively that just a single modal window on a 'start free trial' click.

Possible Invariant metrics:
Number of enrolled free-trial user-ids
a few previous Invariant metrics can be kept to control what happens before enrollments

Possible eval metric: 
Proportion number of payments number of the enrolled user-ids
Proportion of User-ids on a par with hour requirement 5 days into the trial
Proportion of User-ids on a par with hour requirement 10 days into the trial
Proportion of User-ids on a par with hour requirement on last day of trial

We would expect that the to be implemented measures would lead to an increase of the proportion of on-par user-ids in later stages of the 14 days trial.

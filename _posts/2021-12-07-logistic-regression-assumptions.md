---
title: Logistic Regressions Assumption
tags: [Logistic Regression, Algorithm, Assumptions]
style: fill
color: danger
description: I have worked through this notebook created by Kenneth Leung and updated and corrected certain code parts and he merged my pull request into his Original.
---

I have worked through this notebook created by Kenneth Leung and updated and corrected certain code parts and he merged my pull request into his Original.

Original Source: [Kenneth Leung's github repo](https://github.com/kennethleungty/Logistic-Regression-Assumptions)

## Logistic Regression Assumptions

- When the assumptions of logistic regression analysis are not met, problems such as biased coefficient estimates or very large standard errors for the logistic regression coefficients may lead to invalid statistical inferences. 
- In this demo, we go through each key assumption with code examples (on the Titanic dataset)
- Link to TowardsDataScience article: *Coming soon*


### Contents
[**Initial Setup**](#initial-setup)

[**Assumption 1** - Appropriate outcome type](#assumption-outcome) 

[**Assumption 2** - Linearity of independent variables and log odds](#assumption-linearity) 

[**Assumption 3** - No strongly influential outliers](#assumption-outlier)  

[**Assumption 4** - Absence of multicollinearity](#assumption-multicollinearity)  

[**Assumption 5** - Independence of observations](#assumption-independence)  

[**Assumption 6** - Sufficiently large sample size](#assumption-size)  

___

<a name='initial-setup'></a>
### Initial Setup
- Import and pre-process Titanic dataset (suitable example as it is a classification problem)
- Can ignore the details of this segment, as the objective here is just to prepare a dataset for the subsequent assumptions testing
- Reference: https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python


```python
# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import math
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

#### Basic pre-processing


```python
# Import Titanic dataset (train.csv)
df_raw = pd.read_csv('data/train.csv')

# Create categorical variable for traveling alone
df_raw['TravelAlone'] = np.where((df_raw["SibSp"] + df_raw["Parch"])>0, 0, 1).astype('uint8')
df_raw.drop('SibSp', axis=1, inplace=True)
df_raw.drop('Parch', axis=1, inplace=True)
df_raw.drop('PassengerId', axis=1, inplace=True)
df_raw.drop('Name', axis=1, inplace=True)
df_raw.drop('Ticket', axis=1, inplace=True)
df_raw.drop('Cabin', axis=1, inplace=True)
# df_raw.drop('Fare', axis=1, inplace=True)

# Create categorical variables and drop some variables
df_titanic = pd.get_dummies(df_raw, columns=["Pclass","Embarked","Sex"], 
                         drop_first=True) # Remove first variable to prevent collinearity

# Fill NaN (median imputation)
df_titanic["Age"].fillna(df_titanic["Age"].median(skipna=True), inplace=True)

df_titanic.head()
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
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>TravelAlone</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Sex_male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define dependent and independent variables
X_cols = df_titanic.columns.to_list()[1:]
X = df_titanic[X_cols]
y = df_titanic['Survived']

# Add constant
X = sm.add_constant(X, prepend=False)
```

___
<a name='assumption-outcome'></a>
### Assumption 1 - Appropriate outcome type


```python
print(df_titanic['Survived'].nunique())
```

    2
    


```python
df_titanic['Survived'].value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64



- There are only two outcomes (i.e. binary classification of survived or did not survive), so we will be using Binary Logistic Regression (which is the default method we use when we specify family=Binomial in our logit models earlier)
- Other types of Logistic Regression (where outcomes > 2) include:
    - Multinomial Logistic Regression: Target variable has three or more nominal categories such as predicting the type of Wine
    - Ordinal Logistic Regression: Target variable has three or more ordinal categories such as restaurant or product rating from 1 to 5.
- More info: https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

___
<a name='assumption-linearity'></a>
### Assumption 2 - Linearity of independent variables and log odds

#### Box-Tidwell Test

- One of the important assumptions of logistic regression is the linearity of the logit over the continuous covariates. This assumption means that relationships between the continuous predictors and the logit (log odds) is linear. 
- The **Box-Tidwell transformation (test)** can be used to test the linearity in the logit assumption when performing logistic regression.
- It checks whether the logit transform is a linear function of the predictor, effectively adding the non-linear transform of the original predictor as an interaction term to test if this addition made no better prediction.
- A **statistically significant** p-value of the interaction term in the Box-Tidwell transformation means that the linearity assumption is violated
- If one variable is indeed found to be non-linear, then we can resolve it by incorporating higher order polynomial terms for that variable in the regression analysis to capture the non-linearity (e.g. x^2) .- Another solution to this problem is the categorization of the independent variables. That is transforming metric variables to ordinal level and then including them in the model. 

___
- Details on R implementation of Box-Tidwell test in R, please refer to the `Box-Tidwell-Test-in-R.ipynb` notebook
- There is no native Python package to run the Box Tidwell test (unlike in R), so we will be coding the test below manually


```python
# Box Tidwell only works for positive values. Hence, drop values where x = 0
df_titanic_2 = df_titanic.drop(df_titanic[df_titanic.Age == 0].index)
df_titanic_2 = df_titanic_2.drop(df_titanic[df_titanic.Fare == 0].index)

# Export processed df_titanic for separate R notebook: `Box-Tidwell-Test-in-R.ipynb`
# df_titanic_2.to_csv('data/train_processed.csv', index=False)
```


```python
df_titanic_2
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
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>TravelAlone</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Sex_male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>27.0</td>
      <td>13.0000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>19.0</td>
      <td>30.0000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>28.0</td>
      <td>23.4500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>26.0</td>
      <td>30.0000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>32.0</td>
      <td>7.7500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>876 rows × 9 columns</p>
</div>



Logistic Regression with statsmodel - Inclusion of interaction term (logit transform) as part of Box-Tidwell test


```python
df_titanic_lt = df_titanic_2.copy()

# Define continuous variables
continuous_var = ['Age', 'Fare']

# Add logit transform interaction terms (natural log) for continuous variables e.g. Age * Log(Age)
for var in continuous_var:
    df_titanic_lt[f'{var}:Log_{var}'] = df_titanic_lt[var].apply(lambda x: x * np.log(x)) #np.log = natural log

df_titanic_lt.head()
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
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>TravelAlone</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Sex_male</th>
      <th>Age:Log_Age</th>
      <th>Fare:Log_Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>68.002934</td>
      <td>14.362261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>138.228274</td>
      <td>304.141753</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>84.710510</td>
      <td>16.404927</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>124.437182</td>
      <td>210.922595</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>124.437182</td>
      <td>16.789660</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Keep columns related to continuous variables
cols_to_keep = continuous_var + df_titanic_lt.columns.tolist()[-len(continuous_var):]
cols_to_keep
```




    ['Age', 'Fare', 'Age:Log_Age', 'Fare:Log_Fare']




```python
# Redefine independent variables to include interaction terms
X_lt = df_titanic_lt[cols_to_keep]
y_lt = df_titanic_lt['Survived']

# Add constant
X_lt = sm.add_constant(X_lt, prepend=False)
  
# Build model and fit the data (using statsmodel's Logit)
logit_results = GLM(y_lt, X_lt, family=families.Binomial()).fit()

# Display summary results
print(logit_results.summary())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:               Survived   No. Observations:                  876
    Model:                            GLM   Df Residuals:                      871
    Model Family:                Binomial   Df Model:                            4
    Link Function:                  logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -536.19
    Date:                Fri, 03 Dec 2021   Deviance:                       1072.4
    Time:                        12:09:01   Pearson chi2:                     881.
    No. Iterations:                     4   Pseudo R-squ. (CS):             0.1065
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Age              -0.1123      0.058     -1.948      0.051      -0.225       0.001
    Fare              0.0785      0.013      6.057      0.000       0.053       0.104
    Age:Log_Age       0.0218      0.013      1.640      0.101      -0.004       0.048
    Fare:Log_Fare    -0.0119      0.002     -5.251      0.000      -0.016      -0.007
    const            -0.3764      0.402     -0.937      0.349      -1.164       0.411
    =================================================================================
    

- We are interested in the p-values for the logit transformed interaction terms of  `Age:Log_Age` and `Fare:Log_Fare`
- From the summary table above, we can see that the p value for `Fare:Log_Fare` is <0.001, which is statistically significant, whereas `Age:Log_Age` is not
- This means that there is non-linearity in the `Fare` feature, and the assumption has been violated
- We can resolve this by including a polynomial term (e.g. `Fare^2`) to account for the non-linearity

#### Visual Check


```python
# Re-run logistic regression on original set of X and y variables
logit_results = GLM(y, X, family=families.Binomial()).fit()
predicted = logit_results.predict(X)

# Get log odds values
log_odds = np.log(predicted / (1 - predicted))

# Visualize predictor continuous variable vs logit values (Age)
plt.scatter(x = df_titanic['Age'].values, y = log_odds);
plt.show()
```


    
![png](/blog/Logistic_Regression_Assumptions_files/Logistic_Regression_Assumptions_21_0.png )
    


Confirming that there is logit linearity for the `Age` variable (Recall earlier that p value for `Age:Log Age` is 0.101)


```python
# Visualize predictor variable vs logit values for Fare
plt.scatter(x = df_titanic['Fare'].values, y = log_odds);
plt.xlabel("Fare")
plt.ylabel("Log-odds")
plt.show()
```


    
![png](/blog/Logistic_Regression_Assumptions_files/Logistic_Regression_Assumptions_23_0.png)
    


Confirming that there is logit NON-linearity for the `Fare` variable (Recall earlier that p value for `Fare:Log Fare` is <0.001)

___
<a name='assumption-outlier'></a>
### Assumption 3 - No strongly influential outliers
- Influential values are extreme individual data points that can alter the quality of the logistic regression model.
- Cook’s Distance is an estimate of the influence of a data point. It takes into account both the leverage and residual of each observation. Cook’s Distance is a summary of how much a regression model changes when the **i**th observation is removed.
- A general rule of thumb is that any observation with a Cook’s distance greater than 4/n (where n = total observations) is considered to be influential (https://www.statology.org/cooks-distance-python/ and https://www.scikit-yb.org/en/latest/api/regressor/influence.html?highlight=cook#module-yellowbrick.regressor.influence), though there are even more generic cutoff values of >0.5-1.0.
- For outliers, we can use the absolute standardized residuals to identify them (std resid > 3)
- Reference: https://www.statsmodels.org/dev/examples/notebooks/generated/influence_glm_logit.html


```python
# Use GLM method for logreg here so that we can retrieve the influence measures
logit_model = GLM(y, X, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:               Survived   No. Observations:                  891
    Model:                            GLM   Df Residuals:                      882
    Model Family:                Binomial   Df Model:                            8
    Link Function:                  logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -398.95
    Date:                Fri, 03 Dec 2021   Deviance:                       797.91
    Time:                        12:09:02   Pearson chi2:                     933.
    No. Iterations:                     5   Pseudo R-squ. (CS):             0.3536
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Age            -0.0333      0.008     -4.397      0.000      -0.048      -0.018
    Fare            0.0004      0.002      0.164      0.870      -0.004       0.005
    TravelAlone     0.0695      0.196      0.354      0.723      -0.315       0.454
    Pclass_2       -0.9406      0.291     -3.236      0.001      -1.510      -0.371
    Pclass_3       -2.2737      0.289     -7.870      0.000      -2.840      -1.707
    Embarked_Q     -0.0337      0.373     -0.091      0.928      -0.764       0.697
    Embarked_S     -0.5540      0.235     -2.353      0.019      -1.015      -0.093
    Sex_male       -2.5918      0.196    -13.206      0.000      -2.977      -2.207
    const           3.7977      0.456      8.326      0.000       2.904       4.692
    ===============================================================================
    


```python
from scipy import stats

# Get influence measures
influence = logit_results.get_influence()

# Obtain summary df of influence measures
summ_df = influence.summary_frame()

# Filter summary df to Cook distance
diagnosis_df = summ_df.loc[:,['cooks_d']]

# Append absolute standardized residual values
diagnosis_df['std_resid'] = stats.zscore(logit_results.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df.loc[:,'std_resid'].apply(lambda x: np.abs(x))

# Sort by Cook's Distance
diagnosis_df.sort_values("cooks_d", ascending=False)
diagnosis_df
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
      <th>cooks_d</th>
      <th>std_resid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000041</td>
      <td>0.330871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000046</td>
      <td>0.243040</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.001006</td>
      <td>0.866265</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000091</td>
      <td>0.313547</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000017</td>
      <td>0.280754</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0.000292</td>
      <td>0.589428</td>
    </tr>
    <tr>
      <th>887</th>
      <td>0.000049</td>
      <td>0.225427</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0.001102</td>
      <td>1.029623</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0.001417</td>
      <td>0.763489</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0.000133</td>
      <td>0.372182</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 2 columns</p>
</div>




```python
# Set Cook's distance threshold
cook_threshold = 4 / len(df_titanic)
print(f"Threshold for Cook Distance = {cook_threshold}")
```

    Threshold for Cook Distance = 0.004489337822671156
    


```python
# Plot influence measures (Cook's distance)
fig = influence.plot_index(y_var="cooks", threshold=cook_threshold)
plt.axhline(y = cook_threshold, ls="--", color='red')
fig.tight_layout(pad=2)
```


    
![png](/blog/Logistic_Regression_Assumptions_files/Logistic_Regression_Assumptions_29_0.png)
    



```python
# Find number of observations that exceed Cook's distance threshold
outliers = diagnosis_df[diagnosis_df['cooks_d'] > cook_threshold]
prop_outliers = round(100*(len(outliers) / len(df_titanic)),1)
print(f'Proportion of data points that are highly influential = {prop_outliers}%')
```

    Proportion of data points that are highly influential = 6.1%
    


```python
# Find number of observations which are BOTH outlier (std dev > 3) and highly influential
extreme = diagnosis_df[(diagnosis_df['cooks_d'] > cook_threshold) & 
                       (diagnosis_df['std_resid'] > 3)]
prop_extreme = round(100*(len(extreme) / len(df_titanic)),1)
print(f'Proportion of highly influential outliers = {prop_extreme}%')
```

    Proportion of highly influential outliers = 1.3%
    


```python
# Display top 5 most influential outliers
extreme.sort_values("cooks_d", ascending=False).head()
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
      <th>cooks_d</th>
      <th>std_resid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>297</th>
      <td>0.015636</td>
      <td>4.951289</td>
    </tr>
    <tr>
      <th>570</th>
      <td>0.009277</td>
      <td>3.030644</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0.008687</td>
      <td>3.384369</td>
    </tr>
    <tr>
      <th>338</th>
      <td>0.005917</td>
      <td>4.461842</td>
    </tr>
    <tr>
      <th>414</th>
      <td>0.005666</td>
      <td>4.387731</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Deep dive into index 297 (extreme outlier)
df_titanic.iloc[297]
```




    Survived         0.00
    Age              2.00
    Fare           151.55
    TravelAlone      0.00
    Pclass_2         0.00
    Pclass_3         0.00
    Embarked_Q       0.00
    Embarked_S       1.00
    Sex_male         0.00
    Name: 297, dtype: float64



- It is important to note that for data points with relative high Cook's distances, it does **not** automatically mean that it should be immediately removed from the dataset. It is essentially an indicator to highlight which data points are worth looking deeper into, to understand whether they are true anomalies or not
- In practice, an assessment of “large” values is a judgement call based on experience and the particular set of data being analyzed.
- In addition, based on our pre-defined threshold (4/N), only 5% (51/891) of the points are in the outlier zone, which is small as well. The issue comes when there is a significant number of data points classified as outliers.
- The management of outliers is outside the scope of this demo

___
<a name='assumption-multicollinearity'></a>
### Assumption 4 - Absence of multicollinearity


```python
corrMatrix = df_titanic.corr()
plt.subplots(figsize=(9, 5))
sns.heatmap(corrMatrix, annot=True, cmap="RdYlGn")
plt.show()
```


    
![png](/blog/Logistic_Regression_Assumptions_files/Logistic_Regression_Assumptions_36_0.png)
    


- Correlation matrix can be difficult to interpret when there are many independent variables
- Furthermore, not all collinearity problems can be detected by inspection of the correlation matrix: it is possible for collinearity to exist between three or more variables even if no pair of variables has a particularly high correlation. 
- As such, we use Variance Inflation Factor (VIF) instead

#### Variance Inflation Factor (VIF)


```python
# Use variance inflation factor to identify any significant multi-collinearity
def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif)

calc_vif(df_titanic)
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
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Survived</td>
      <td>1.944148</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>5.005814</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fare</td>
      <td>1.793238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TravelAlone</td>
      <td>3.030957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_2</td>
      <td>1.968630</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_3</td>
      <td>3.524367</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Embarked_Q</td>
      <td>1.591633</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Embarked_S</td>
      <td>4.795192</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sex_male</td>
      <td>3.708845</td>
    </tr>
  </tbody>
</table>
</div>



- The threshold for VIF is usually 5 (i.e. values above 5 means there is presence of multicollinearity)
- Since all the variables have VIF <5,  it means that there is no multicollinearity, and this assumption is satisfied
- Let's have a look at the situation where we did **not** drop the first variable upon getting dummies:


```python
# Avoid dropping first variables upon get_dummies
df_test = pd.get_dummies(df_raw, columns=["Pclass","Embarked","Sex"], 
                           drop_first=False)
df_test.drop('Sex_female', axis=1, inplace=True)
df_test["Age"].fillna(df_test["Age"].median(skipna=True), inplace=True)
calc_vif(df_test)
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
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Survived</td>
      <td>1.636129</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>1.247705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fare</td>
      <td>1.690089</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TravelAlone</td>
      <td>1.223353</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass_1</td>
      <td>117.152079</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pclass_2</td>
      <td>99.102382</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pclass_3</td>
      <td>260.025558</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Embarked_C</td>
      <td>69.936806</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Embarked_Q</td>
      <td>36.792002</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embarked_S</td>
      <td>91.326578</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sex_male</td>
      <td>1.539417</td>
    </tr>
  </tbody>
</table>
</div>



- From the above results, we can see that there are numerous VIF values way above the threshold of 5 when we do not drop at least 1 category from the dummy categories we generated
- This is a clear sign of multicollinearity

___
<a name='assumption-independence'></a>
### Assumption 5 - Independence of observations
- Error terms need to be independent. That is that the data-points should not be from any dependent samples design, e.g., before-after measurements, or matched pairings. 


```python
# Setup logistic regression model (using GLM method so that we can retrieve residuals)
logit_model = GLM(y, X, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:               Survived   No. Observations:                  891
    Model:                            GLM   Df Residuals:                      882
    Model Family:                Binomial   Df Model:                            8
    Link Function:                  logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -398.95
    Date:                Fri, 03 Dec 2021   Deviance:                       797.91
    Time:                        12:09:03   Pearson chi2:                     933.
    No. Iterations:                     5   Pseudo R-squ. (CS):             0.3536
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Age            -0.0333      0.008     -4.397      0.000      -0.048      -0.018
    Fare            0.0004      0.002      0.164      0.870      -0.004       0.005
    TravelAlone     0.0695      0.196      0.354      0.723      -0.315       0.454
    Pclass_2       -0.9406      0.291     -3.236      0.001      -1.510      -0.371
    Pclass_3       -2.2737      0.289     -7.870      0.000      -2.840      -1.707
    Embarked_Q     -0.0337      0.373     -0.091      0.928      -0.764       0.697
    Embarked_S     -0.5540      0.235     -2.353      0.019      -1.015      -0.093
    Sex_male       -2.5918      0.196    -13.206      0.000      -2.977      -2.207
    const           3.7977      0.456      8.326      0.000       2.904       4.692
    ===============================================================================
    

#### Check residuals series


```python
# Generate residual series plot
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot",
    xlabel="Index Number", ylabel="Deviance Residuals")

# ax.plot(df_titanic_2.index.tolist(), stats.zscore(logit_results.resid_pearson))
ax.plot(df_titanic.index.tolist(), stats.zscore(logit_results.resid_deviance))
plt.axhline(y = 0, ls="--", color='red');
```


    
![png](/blog/Logistic_Regression_Assumptions_files/Logistic_Regression_Assumptions_46_0.png)
    


- From the above Deviance residuals versus index number plot, we can see that the assumption of independence of errors is satisfied

#### Further investigation on residual dependence plots (Optional)
- This is additional investigation. The above check on residuals series (based on index numbers) is sufficient
- Reference: https://freakonometrics.hypotheses.org/8210


```python
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(
    111,
    title="Residual Dependence Plot",
    xlabel="Fitted Values",
    ylabel="Pearson Residuals",
)
# ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_pearson))
ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_deviance))
ax.axis("tight")
ax.plot([0.0, 1.0], [0.0, 0.0], "k-");
```


    
![png](/blog/Logistic_Regression_Assumptions_files/Logistic_Regression_Assumptions_49_0.png)
    


#### Add a Locally Weighted Scatterplot Smoothing (LOWESS) line to better visualize independence


```python
# Setup LOWESS function
lowess = sm.nonparametric.lowess

# Get y-values from LOWESS (set return_sorted=False)
y_hat_lowess = lowess(logit_results.resid_pearson, logit_results.mu, 
                      return_sorted = False,
                      frac=2/3)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111,
    title="Residual Dependence Plot",
    xlabel="Fitted Values",
    ylabel="Pearson Residuals",
)
# ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_pearson))
ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_deviance))
ax.scatter(logit_results.mu, y_hat_lowess)
ax.axis("tight")
ax.plot([0.0, 1.0], [0.0, 0.0], "k-");
```


    
![png](/blog/Logistic_Regression_Assumptions_files/Logistic_Regression_Assumptions_51_0.png)
    


___
<a name='assumption-size'></a>
### Assumption 6 - Sufficiently large sample size


```python
# Find total number of observations
len(df_titanic)
```




    891




```python
# Get value counts for independent variables (mainly focus on categorical)
for col in df_titanic.columns.to_list()[1:]:
    if df_titanic.dtypes[col] == 'uint8': # Keep categorical variables only
        print(df_titanic[col].value_counts())
```

    1    537
    0    354
    Name: TravelAlone, dtype: int64
    0    707
    1    184
    Name: Pclass_2, dtype: int64
    1    491
    0    400
    Name: Pclass_3, dtype: int64
    0    814
    1     77
    Name: Embarked_Q, dtype: int64
    1    644
    0    247
    Name: Embarked_S, dtype: int64
    1    577
    0    314
    Name: Sex_male, dtype: int64
    

- Rule of thumb is to have at least 10-20 instances of the least frequent outcome for each predictor variable in your model
- From the value counts above, we can see that this assumption is satisfied

- Another rule of thumb is to have at least 500 observations in the entire dataset
- Overall, we have 891 observations, which is a decent dataset size to work with

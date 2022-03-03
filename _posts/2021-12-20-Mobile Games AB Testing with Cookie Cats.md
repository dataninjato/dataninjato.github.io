---
title: Mobile Games AB Testing with Cookie Cats
tags: [AB-testing]
style: fill
color: secondary
comments: true
description: Where should the gates be placed? Initially the first gate was placed at level 30, but in this notebook we're going to analyze an AB-test where we moved the first gate in Cookie Cats from level 30 to level 40. In particular, we will look at the impact on player retention.
---

## 1. Of cats and cookies
<p>Cookie Cats is a hugely popular mobile puzzle game developed by Tactile Entertainment. It's a classic "connect three"-style puzzle game where the player must connect tiles of the same color to clear the board and win the level. It also features singing cats. We're not kidding! Check out this short demo:</p>
<p></p>
<p>As players progress through the levels of the game, they will occasionally encounter gates that force them to wait a non-trivial amount of time or make an in-app purchase to progress. In addition to driving in-app purchases, these gates serve the important purpose of giving players an enforced break from playing the game, hopefully resulting in that the player's enjoyment of the game being increased and prolonged.</p>

<img src='cc_gates.png'>

<p>But where should the gates be placed? Initially the first gate was placed at level 30, but in this notebook we're going to analyze an AB-test where we moved the first gate in Cookie Cats from level 30 to level 40. In particular, we will look at the impact on player retention. But before we get to that, a key step before undertaking any analysis is understanding the data. So let's load it in and take a look!</p>


```python
# Importing pandas
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')

# Reading in the data
df = pd.read_csv('datasets/cookie_cats.csv')

# Showing the first few rows
df.head(5)
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
      <th>userid</th>
      <th>version</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>116</td>
      <td>gate_30</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>337</td>
      <td>gate_30</td>
      <td>38</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>377</td>
      <td>gate_40</td>
      <td>165</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>483</td>
      <td>gate_40</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>488</td>
      <td>gate_40</td>
      <td>179</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## 2. The AB-test data
<p>The data we have is from 90,189 players that installed the game while the AB-test was running. The variables are:</p>
<ul>
<li><code>userid</code> - a unique number that identifies each player.</li>
<li><code>version</code> - whether the player was put in the control group (<code>gate_30</code> - a gate at level 30) or the group with the moved gate (<code>gate_40</code> - a gate at level 40).</li>
<li><code>sum_gamerounds</code> - the number of game rounds played by the player during the first 14 days after install.</li>
<li><code>retention_1</code> - did the player come back and play <strong>1 day</strong> after installing?</li>
<li><code>retention_7</code> - did the player come back and play <strong>7 days</strong> after installing?</li>
</ul>
<p>When a player installed the game, he or she was randomly assigned to either <code>gate_30</code> or <code>gate_40</code>. As a sanity check, let's see if there are roughly the same number of players in each AB group. </p>


```python
# Counting the number of players in each AB group.
df.groupby('version').count()

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
      <th>userid</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gate_30</th>
      <td>44700</td>
      <td>44700</td>
      <td>44700</td>
      <td>44700</td>
    </tr>
    <tr>
      <th>gate_40</th>
      <td>45489</td>
      <td>45489</td>
      <td>45489</td>
      <td>45489</td>
    </tr>
  </tbody>
</table>
</div>



## 3. The distribution of game rounds

<p>It looks like there is roughly the same number of players in each group, nice!</p>
<p>The focus of this analysis will be on how the gate placement affects player retention, but just for fun: Let's plot the distribution of the number of game rounds players played during their first week playing the game.</p>


```python
# Counting the number of players for each number of gamerounds 
plot_df = df.groupby('sum_gamerounds').userid.count()

# Plotting the distribution of players that played 0 to 100 game rounds
ax = plot_df.head(100).plot(x='sum_gamerounds', y='number users')
ax.set_xlabel("Sum Gamerounds")
ax.set_ylabel("number Users")
```




    Text(0, 0.5, 'number Users')




    
![png](/blog/notebook%20AB%20Testing%20with%20Cookie%20Cats_files/notebook%20AB%20Testing%20with%20Cookie%20Cats_6_1.png)
    



```python
plot_df
```




    sum_gamerounds
    0        3994
    1        5538
    2        4606
    3        3958
    4        3629
             ... 
    2294        1
    2438        1
    2640        1
    2961        1
    49854       1
    Name: userid, Length: 942, dtype: int64



## 4. Overall 1-day retention
<p>In the plot above we can see that some players install the game but then never play it (0 game rounds), some players just play a couple of game rounds in their first week, and some get really hooked!</p>
<p>What we want is for players to like the game and to get hooked. A common metric in the video gaming industry for how fun and engaging a game is <em>1-day retention</em>: The percentage of players that comes back and plays the game <em>one day</em> after they have installed it.  The higher 1-day retention is, the easier it is to retain players and build a large player base. </p>
<p>As a first step, let's look at what 1-day retention is overall.</p>


```python
# The % of users that came back the day after they installed

df[df['retention_1'] == True].retention_1.count() / df.retention_1.count()
```




    0.4452095044850259



## 5. 1-day retention by AB-group

<p>So, a little less than half of the players come back one day after installing the game. Now that we have a benchmark, let's look at how 1-day retention differs between the two AB-groups.</p>


```python
# Calculating 1-day retention for control and treatment group

returning_users = df['retention_1'] == True
ret1_per_group = df[returning_users].groupby('version').retention_1.count() / df.groupby('version').retention_1.count()
ret1_per_group
```




    version
    gate_30    0.448188
    gate_40    0.442283
    Name: retention_1, dtype: float64



## 6. Should we be confident in the difference?
<p>It appears that there was a slight decrease in 1-day retention when the gate was moved to level 40 (44.2%) compared to the control when it was at level 30 (44.8%). It's a small change, but even small changes in retention can have a large impact. But while we are certain of the difference in the data, how certain should we be that a gate at level 40 will be worse in the future?</p>
<p>There are a couple of ways we can get at the certainty of these retention numbers. Here we will use bootstrapping: We will repeatedly re-sample our dataset (with replacement) and calculate 1-day retention for those samples. The variation in 1-day retention will give us an indication of how uncertain the retention numbers are.</p>


```python
%%time
# Creating an list with bootstrapped means for each AB-group
np.random.seed(1)

boot_1d = []
iterations=1000
for i in range(iterations):
    boot_mean = df.sample(frac=1, replace=True).groupby('version').retention_1.mean()
    boot_1d.append(boot_mean)
    
# Transforming the list to a DataFrame
boot_1d = pd.DataFrame(boot_1d)
    
# A Kernel Density Estimate plot of the bootstrap distributions
#boot_1d.boxplot(column=["gate_30", "gate_40"])
boot_1d.plot.kde();
plt.show()
sns.set(rc={'figure.figsize':(6,6)})
ax = sns.boxplot(data=boot_1d)
ax = sns.swarmplot(data=boot_1d, color=".25", size=2)
ax.set_ylabel("sample means")
```


    
![png](/blog/notebook%20AB%20Testing%20with%20Cookie%20Cats_files/notebook%20AB%20Testing%20with%20Cookie%20Cats_13_0.png)
    


    Wall time: 9.87 s
    




    Text(0, 0.5, 'sample means')




    
![png](/blog/notebook%20AB%20Testing%20with%20Cookie%20Cats_files/notebook%20AB%20Testing%20with%20Cookie%20Cats_13_3.png)
    


## 7. Zooming in on the difference
<p>These two distributions above represent the bootstrap uncertainty over what the underlying 1-day retention could be for the two AB-groups. Just eyeballing this plot, we can see that there seems to be some evidence of a difference, albeit small. Let's zoom in on the difference in 1-day retention</p>
<p>(<em>Note that in this notebook we have limited the number of bootstrap replication to 500 to keep the calculations quick. In "production" we would likely increase this to a much larger number, say, 10 000.</em>)</p>


```python
# Adding a column with the % difference between the two AB-groups
boot_1d['diff'] = (boot_1d['gate_30'] - boot_1d['gate_40']) / boot_1d['gate_40'] * 100

# Ploting the bootstrap % difference
sns.set(rc={'figure.figsize':(8,5)})
ax = boot_1d['diff'].plot.kde(lw = 2,)
ax.set_xlabel("difference in percent")
ax.axvline(0, color ='green', lw = 3, alpha = 0.5)
ax.text(0.1,0.01,'H0')
plt.show()
```


    
![png](/blog/notebook%20AB%20Testing%20with%20Cookie%20Cats_files/notebook%20AB%20Testing%20with%20Cookie%20Cats_15_0.png)
    


## 8. The probability of a difference

<p>From this chart, we can see that the most likely % difference is around 1% - 2%, and that most of the distribution is above 0%, in favor of a gate at level 30. But what is the <em>probability</em> that the difference is above 0%? Let's calculate that as well.</p>


```python
# Calculating the probability that 1-day retention is greater when the gate is at level 30
prob = (boot_1d['diff'] > 0).sum() / len(boot_1d['diff'])

# Pretty printing the probability
'{:.1%}'.format(prob)
```




    '96.5%'



## 9. 7-day retention by AB-group
<p>The bootstrap analysis tells us that there is a high probability that 1-day retention is better when the gate is at level 30. However, since players have only been playing the game for one day, it is likely that most players haven't reached level 30 yet. That is, many players won't have been affected by the gate, even if it's as early as level 30. </p>
<p>But after having played for a week, more players should have reached level 40, and therefore it makes sense to also look at 7-day retention. That is: What percentage of the people that installed the game also showed up a week later to play the game again.</p>
<p>Let's start by calculating 7-day retention for the two AB-groups.</p>


```python
# Calculating 7-day retention for both AB-groups
df[df['retention_7'] ==  True].groupby('version').retention_7.count() / df.groupby('version').retention_7.count()
```




    version
    gate_30    0.190201
    gate_40    0.182000
    Name: retention_7, dtype: float64



## 10. Bootstrapping the difference again
<p>Like with 1-day retention, we see that 7-day retention is slightly lower (18.2%) when the gate is at level 40 than when the gate is at level 30 (19.0%). This difference is also larger than for 1-day retention, presumably because more players have had time to hit the first gate. We also see that the <em>overall</em> 7-day retention is lower than the <em>overall</em> 1-day retention; fewer people play a game a week after installing than a day after installing.</p>
<p>But as before, let's use bootstrap analysis to figure out how certain we should be of the difference between the AB-groups.</p>


```python
%%time
iterations=1000

# Creating a list with bootstrapped means for each AB-group
boot_7d = []
for i in range(iterations):
    boot_mean = df.sample(frac=1, replace=True).groupby('version').retention_7.mean()
    boot_7d.append(boot_mean)
    
# Transforming the list to a DataFrame
boot_7d = pd.DataFrame(boot_7d)
    
# A Kernel Density Estimate plot of the bootstrap distributions
#print(boot_7d.boxplot(column=["gate_30", "gate_40"]))
sns.set(rc={'figure.figsize':(6,6)})
ax = sns.boxplot(data=boot_7d)
ax.set_ylabel("sample means")
plt.show()

# Ploting the bootstrap % difference
boot_7d['diff'] = (boot_7d['gate_30'] - boot_7d['gate_40']) / boot_7d['gate_40'] * 100

ax2 = boot_7d['diff'].plot.kde(lw=2);
ax2.axvline(0, color ='green', lw = 3, alpha = 0.5)
ax2.text(0.1,0.05,'H0')
ax2.set_xlabel("difference in percent")


# Calculating the probability that 7-day retention is greater when the gate is at level 30
prob = (boot_7d['diff'] > 0).sum() / len(boot_7d['diff'])

# Pretty printing the probability
'{:.3%}'.format(prob)
```


    
![png](/blog/notebook%20AB%20Testing%20with%20Cookie%20Cats_files/notebook%20AB%20Testing%20with%20Cookie%20Cats_21_0.png)
    


    Wall time: 8.75 s
    




    '100.000%'




    
![png](/blog/notebook%20AB%20Testing%20with%20Cookie%20Cats_files/notebook%20AB%20Testing%20with%20Cookie%20Cats_21_3.png)
    


## 11.  The conclusion
<p>The bootstrap result tells us that there is strong evidence that 7-day retention is higher when the gate is at level 30 than when it is at level 40. The conclusion is: If we want to keep retention high — both 1-day and 7-day retention — we should <strong>not</strong> move the gate from level 30 to level 40. There are, of course, other metrics we could look at, like the number of game rounds played or how much in-game purchases are made by the two AB-groups. But retention <em>is</em> one of the most important metrics. If we don't retain our player base, it doesn't matter how much money they spend in-game.</p>

<p>So, why is retention higher when the gate is positioned earlier? One could expect the opposite: The later the obstacle, the longer people are going to engage with the game. But this is not what the data tells us. The theory of <em>hedonic adaptation</em> can give one explanation for this. In short, hedonic adaptation is the tendency for people to get less and less enjoyment out of a fun activity over time if that activity is undertaken continuously. By forcing players to take a break when they reach a gate, their enjoyment of the game is prolonged. But when the gate is moved to level 40, fewer players make it far enough, and they are more likely to quit the game because they simply got bored of it. </p>


```python
# So, given the data and the bootstrap analysis
# Should we move the gate from level 30 to level 40 ?
move_to_level_40 = False

#!tar chvfz notebook.tar.gz *
```

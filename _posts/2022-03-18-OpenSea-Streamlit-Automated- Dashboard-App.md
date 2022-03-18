---
title: OpenSea API and Scrape Explorer Streamlit Automated Dashboard App
tags: [Streamlit, Dashboard, API, Scraping, Analysis Tool, Heroku]
style: fill
color: yellow
description: Self Servicing Demo of an app that provides real added value and high reusability for more than one user only. A Data Analyst can support and reach a wide audience of users, who would like to self service and maybe ad-hoc require a piece of information right away.
---

Instead of opting for another Exploratory Data Analysis App for Machine Learning Models or its Data, that does not really require code packaged with Streamlit and can be done in regular scripts as well, I decided to have an app that actually provides added value and high reusability for more than one user only. A Data Analyst can support and reach a wide audience of users, who would like to self service and maybe ad-hoc require a piece of information right away.

The Strength of Streamlit Dashboards is the agility. You can really do rapid prototyping, invest a few hours and get results and a working prototype, which can be hosted directly from a github repository. 

Streamlit is also great in being able to integrate web-serviced dashboards of PowerBI or Tableau directly into the App (noted on the ToDo list).

This Streamlit app is forked from here [Github Repo](https://github.com/hackingthemarkets/opensea-nft-api-tutorial)

and modified and extended by 
1. Rarity Rankings distributions and individual tokens. Rarity Ranks based on Average Jaccard Distance previously implemented in the other blog post [Non Fungible Token (NFT) Rarity Analysis with Jaccard Distance](https://dataninjato.github.io/blog/non-fungible-token-nft-rarity-analysis-with-jaccard-distance) (Data is pulled from precomputed Files)
2. Collection Market Stats (Data pulled from Opensea API)
3. Collection Market in Detail by looking at individual orders and their price value positions (Data scraped live from Opensea collection urls)


Check out the app here: Hosted 24/7 on Heroku: [https://opensea-nft-api-scraper-stream.herokuapp.com/](https://opensea-nft-api-scraper-stream.herokuapp.com/)

![Expensive NFTs](/blog/expensive.png "View: Expensive NFT")

# 1. Rarity Rankings distributions and individual tokens

![Rarity Ranks](/blog/rarity-rank.png "View: Rarity Ranks")


```python
# Excerpt from Streamlit App

if endpoint == 'Rarity Ranks':
    import seaborn as sns
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import percentileofscore
    sns.set_style('darkgrid')
    
    endpoint2 = st.sidebar.selectbox("Collection", [
                                                    'richsimmonsreflections',
                                                    're-genz',
                                                    'official-surreals',
                                                    'galaxyeggs9999'
                                                    ]
                                    )
    token_id = st.sidebar.text_input("Token ID")
    
    collection_slug = endpoint2

    # load rank data
    score = pd.read_pickle('score_' + collection_slug + '.pkl')
    score['rank'] = score['score'].rank(method='min', ascending=False)
    score['rank'] = score['rank'].astype(int)
    
    fig = plt.figure(figsize=(10, 6))
    
    score.score.hist(bins=20, alpha=0.6, label="Count")
    plt.xlabel('Scores')
    plt.ylabel('Number of Tokens')
    plt.title('Token Score Distribution')
    st.pyplot(fig) 
    
    if token_id:
        try:
            token = int(token_id)

            st.write('# Token ID #',token_id, 'ranks on rank number: ', score.loc[[token]]['rank'].to_numpy()[0], ' of max: ', score.shape[0])

            calc_percentile = percentileofscore(score['score'], score.loc[token]['score'])   
            st.write('# Token ', token, ' represents the ', round(calc_percentile,2),' Percentile of this average jaccard distance distribution')
        except:
            print('error')
```

The precomputed Jaccard Distance Average as Scores are read from file and then ranked based on descending scores of 0 to 1.

All Scores are plotted into a Histogram to see the collection distribution as a whole.

If a individual token is queried in the sideboard, its rank and percentile is then given for the user to assess its rarity.

Further things that could be done: 

- interactive charting with e.g. plotly to show where the token score is located at in the distribution
- trait facetting

# 2. Collection Market Stats (Data pulled from Opensea API)

![Stats](/blog/stats.png "View: Stats")


```python
if endpoint == 'Stats':
    
    endpoint2 = st.sidebar.selectbox("Collection", ['galaxyeggs9999',
                                                    'richsimmonsreflections',
                                                    're-genz',
                                                    'official-surreals',
                                                    'officialtimelessapeclub',
                                                    'geisha-tea-house',
                                                    'the-crypto-chicks',
                                                    ]
                                    )
    id = endpoint2
    url = "https://api.opensea.io/api/v1/collection/{0}/stats".format(id)

    r = requests.get(url)
    
    st.header(id)
      
    st.metric(label="floor price", value=r.json()["stats"]["floor_price"], 
                delta=round(r.json()["stats"]["floor_price"]*-0.05,4),  # delta just a dummy as no previous data is saved
                )

    st.metric(label="24h Volume", value=r.json()["stats"]["one_day_volume"], 
                delta=round(r.json()["stats"]["one_day_volume"]*0.05,4),  # delta just a dummy as no previous data is saved
                )
    st.write(f"Number of sales in 24h:", int(r.json()["stats"]["one_day_sales"]))
    
    st.write(f"Number of sales in last 30 days", int(r.json()["stats"]["thirty_day_sales"]))
    st.write(f"24h AVG price:", round(r.json()["stats"]["one_day_average_price"],4),f"ETH")
    st.write(f"Unique owners:", r.json()["stats"]["num_owners"],f"/", r.json()["stats"]["count"], f"ratio:", round(r.json()["stats"]["num_owners"]/r.json()["stats"]["count"]*100,2), f"%")
    # pace requests
    pause = round(random.uniform(0.8, 1.4),2)
    time.sleep(pause)
```

OpenSea Collection Stats are retrieved from its API including popular metrics such as ***floor price*** and ***volume**.

Calls are paced in order to avoid getting throttled by the OpenSea API.

Further things that could be done: 

- automated querying of these stats and saving it to a cloud database to have meaningful deltas displayed in the metrics
- analysis and charting of this data history

# 3. Collection Market in Detail by looking at individual orders and their price value positions (Data scraped live from Opensea collection urls)

![Price Rank Product](/blog/price-rank-product.png "View: Price Rank Product")


```python
if endpoint == 'Price Rank Product':
    
    collection_slug = 'richsimmonsreflections'
    import re
    import requests
    from bs4 import BeautifulSoup as bs

    def checkmarket(url, collection_slug=collection_slug):
        
        url_stats = "https://api.opensea.io/api/v1/collection/" + collection_slug + "/stats"
        r = requests.get(url_stats)
        fp = r.json()["stats"]["floor_price"]
        r = scraper.get(url).text

        market = pd.DataFrame(columns=['token_id', 'price', 'rank', 'price_rank_product', 'diff_to_fp', 'dfpp'])

        html_code = bs(r, 'html.parser')
        cards = html_code.find_all('article', class_=re.compile('^AssetSearchList--asset'))
        
        # load rank data
        score = pd.read_pickle('score_' + collection_slug + '.pkl')
        score['rank'] = score['score'].rank(method='min', ascending=False)
        score['rank'] = score['rank'].astype(int)
        
        for c in cards:
            
            price = c.find('div', class_=re.compile('^Price--amount')).getText(strip=True).split("#",1)[0] if c.find('div', class_=re.compile('^Price--amount')) else "none"
            token = c.find('div', class_=re.compile('^AssetCardFooter--name')).getText(strip=True).split("#",1)[0] if c.find('div', class_=re.compile('^AssetCardFooter--name')) else "none"
            
            try:
                token = int(token)  
            except:
                continue
            else:
                token_rank = score.loc[score.token_id == token]['rank'].to_numpy()

                if len(token_rank) == 1:
                    try:
                        grade = round(float(price) * token_rank[0],1)
                    except:
                        print(price, token_rank)
                        grade = float(0.001) # dummy              
                    try:
                        diff_to_fp = float(price) - float(fp)
                        dfpp = round(float(diff_to_fp) * token_rank[0],1)
                    except:
                        dfpp = float(0.001) # dummy
                else: #lazy loaded cards
                    token_rank = str(token_rank)
                    grade = float(0.0001) # dummy
                    dfpp = float(0.0001) # dummy
                market.loc[len(market.index)] = [token, price, token_rank, grade, diff_to_fp, dfpp]
        return market, fp
                                                
    endpoint4 = st.sidebar.selectbox("Filter", ['Legendaries',
                                                    'Signatures',
                                                    'Delist'
                                                    ]
                                    )

    filterurls = {'Legendaries': 'https://opensea.io/collection/richsimmonsreflections?search[sortAscending]=true&search[sortBy]=PRICE&search[stringTraits][0][name]=SHINIES&search[stringTraits][0][values][0]=BANKSY%20%28WITH%20SIGNATURE%29%20BLACK%20REFLECTIONS&search[stringTraits][0][values][1]=BANKSY%20GOLD%20REFLECTIONS&search[stringTraits][0][values][2]=BANKSY%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][3]=BANKSY%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][4]=BASQUIAT%20%28WITH%20SIGNATURE%29%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][5]=BASQUIAT%20BLACK%20REFLECTIONS&search[stringTraits][0][values][6]=BASQUIAT%20GOLD%20REFLECTIONS&search[stringTraits][0][values][7]=BASQUIAT%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][8]=BOTICELLI%20%28WITH%20SIGNATURE%29%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][9]=BOTICELLI%20BLACK%20REFLECTIONS&search[stringTraits][0][values][10]=BOTICELLI%20GOLD%20REFLECTIONS&search[stringTraits][0][values][11]=BOTICELLI%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][12]=DA%20VINCI%20%28WITH%20SIGNATURE%29%20GOLD%20REFLECTIONS&search[stringTraits][0][values][13]=DA%20VINCI%20BLACK%20REFLECTIONS&search[stringTraits][0][values][14]=DA%20VINCI%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][15]=DA%20VINCI%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][16]=DALI%20%28WITH%20SIGNATURE%29%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][17]=DALI%20BLACK%20REFLECTIONS&search[stringTraits][0][values][18]=DALI%20GOLD%20REFLECTIONS&search[stringTraits][0][values][19]=DALI%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][20]=HARING%20%28WITH%20SIGNATURE%29%20BLACK%20REFLECTIONS&search[stringTraits][0][values][21]=HARING%20GOLD%20REFLECTIONS&search[stringTraits][0][values][22]=HARING%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][23]=HARING%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][24]=HIRST%20%28WITH%20SIGNATURE%29%20GOLD%20REFLECTIONS&search[stringTraits][0][values][25]=HIRST%20BLACK%20REFLECTIONS&search[stringTraits][0][values][26]=HIRST%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][27]=HIRST%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][28]=LICHTENSTEIN%20%28WITH%20SIGNATURE%29%20GOLD%20REFLECTIONS&search[stringTraits][0][values][29]=LICHTENSTEIN%20BLACK%20REFLECTIONS&search[stringTraits][0][values][30]=LICHTENSTEIN%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][31]=LICHTENSTEIN%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][32]=MUNCH%20%28WITH%20SIGNATURE%29%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][33]=MUNCH%20BLACK%20REFLECTIONS&search[stringTraits][0][values][34]=MUNCH%20GOLD%20REFLECTIONS&search[stringTraits][0][values][35]=MUNCH%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][36]=PICASSO%20%28WITH%20SIGNATURE%29%20GOLD%20REFLECTIONS&search[stringTraits][0][values][37]=PICASSO%20BLACK%20REFLECTIONS&search[stringTraits][0][values][38]=PICASSO%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][39]=PICASSO%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][40]=SIMMONS%20%28WITH%20SIGNATURE%29%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][41]=SIMMONS%20BLACK%20REFLECTIONS&search[stringTraits][0][values][42]=SIMMONS%20GOLD%20REFLECTIONS&search[stringTraits][0][values][43]=SIMMONS%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][44]=VAN%20GOGH%20BLACK%20REFLECTIONS&search[stringTraits][0][values][45]=VAN%20GOGH%20GOLD%20REFLECTIONS&search[stringTraits][0][values][46]=VAN%20GOGH%20HOLOGRAPHIC%20REFLECTIONS&search[stringTraits][0][values][47]=VAN%20GOGH%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][48]=WARHOL%20%28WITH%20SIGNATURE%29%20RAINBOW%20REFLECTIONS&search[stringTraits][0][values][49]=WARHOL%20BLACK%20REFLECTIONS&search[stringTraits][0][values][50]=WARHOL%20GOLD%20REFLECTIONS&search[stringTraits][0][values][51]=WARHOL%20HOLOGRAPHIC%20REFLECTIONS&search[toggles][0]=BUY_NOW',
    'Signatures':'https://opensea.io/collection/richsimmonsreflections?search[sortAscending]=true&search[sortBy]=PRICE&search[stringTraits][0][name]=PAINTINGS&search[stringTraits][0][values][0]=LICHTENSTEIN%20SIGNATURE&search[stringTraits][0][values][1]=BANKSY%20SIGNATURE&search[stringTraits][0][values][2]=BOTICELLI%20SIGNATURE&search[stringTraits][0][values][3]=SIMMONS%20SIGNATURE&search[stringTraits][0][values][4]=HARING%20SIGNATURE&search[stringTraits][0][values][5]=HIRST%20SIGNATURE&search[stringTraits][0][values][6]=MUNCH%20SIGNATURE&search[stringTraits][0][values][7]=DA%20VINCI%20SIGNATURE&search[stringTraits][0][values][8]=VAN%20GOGH%20SIGNATURE&search[stringTraits][0][values][9]=WARHOL%20SIGNATURE&search[stringTraits][0][values][10]=BASQUIAT%20SIGNATURE&search[stringTraits][0][values][11]=DALI%20SIGNATURE&search[stringTraits][0][values][12]=PICASSO%20SIGNATURE',
    'Delist':'https://opensea.io/collection/richsimmonsreflections?search[sortAscending]=true&search[sortBy]=PRICE&search[toggles][0]=BUY_NOW' 
    }
    
    url = filterurls[endpoint4]
    m1, fp = checkmarket(url)[0:32] 
    m1 = m1.drop_duplicates(subset=['token_id', 'price'])
    m1 = m1.sort_values('price_rank_product')
    
    def highlight_lessthan(x):
        if x.price_rank_product < 10:
            return ['background-color: yellow']*6
        else:
            return ['background-color: white']*6
    m1 = m1.style.apply(highlight_lessthan, axis=1)
    
    st.header(collection_slug + ' - ' + endpoint4)
    st.write('floor price: ', fp)
    st.dataframe(m1, height=750)
    
    # pace scrape requests 
    pause = round(random.uniform(0.8, 1.4),2)
    time.sleep(pause)
```

The OpenSea API and its 'Retrieving orders' get method is pretty poor in retrieving relevant orders on a given collection market as all assets are supposed to be systematically checked in 50 pieces chunks. Often enough only few orders are present. This requires often 50 calls or more for a whole collection. In contrast you can get much more information with a single or a few scrapes combined of the Opensea market urls, which is much less expensive. So Scraping it is!

After we have managed to overcome the cloudflare's protection by sending headers with help of the cloudscrapers package, we use beautifulsoup4 to facilitate the actual scraping work and to retrieve token ids and buy now prices listed. We need to confine ourselves to the first 32 listings shown with an OpenSea url, as additional orders are lazy loaded by javascript and would require another approach with selenium and chromedriver browserengine. This is hardly possible to host that with the heroku free hosting plan however.

Next to token ids and prices we retrieve the precomputed rarity ranks of the individual tokens and all save into a single market dataframe, which is displayed and sorted for ascending price rank product (a measure to see identify high value of low rank and low price orders listed).

Further things to do: 

- charting orders two dimensionally by score and log price for visual assessment

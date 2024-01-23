---
title: Dockerize Python market data scraping app with postgresql db
tags: [Docker, Python, SQL, Postgres, Scraping, Server, Power BI]
style: border
color: primary
description: Data Analytics requires conistent error free data generation and therefore our Python scraping app will run in a docker container environment which is isolated from host issues and retains the data at the same time between runs for later analysis.
---
<style type="text/css">
 .markdown-body img:not(.emoji) {
    display: block;
    max-width: 1349px; !important
}
</style>

Developing a python app is one thing but having it running on schedule regularly requires it to be more flexible and portable, so that it can be run on any machine in a standardized environment that enables error free consistent runtimes. I like to have complete time series data and having frequently missing time periods is not acceptable.

The solution to this data analytics requirement is to have this app run in a docker container environment which is isolated from host issues. Combined with a docker persitent volume for the postgresql database data to be stored fits the bill and would allow me to even remove a local postgresql installation, if this docker app today was not supposed to be run on a mini server unit 24/7 anyway.

For that purpose we slightly adapt a standard docker recipe in form of a docker `compose.yml` file for this app:


```python
services:
  server:
    build:
      context: .
    ports:
      - 5000:5000
    environment:
      - POSTGRES_PASSWORD=mysecretpassword
    depends_on:
      db:
        condition: service_healthy
  db:
    image: postgres
    restart: always
    user: postgres
    secrets:
      - db-password
    volumes:
      - db-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=barter
      - POSTGRES_PASSWORD_FILE=/run/secrets/db-password
    ports:
      - "5432:5432"
    expose:
      - 5432
    healthcheck:
      test: [ "CMD", "pg_isready" ]
      interval: 10s
      timeout: 5s
      retries: 5
volumes:
  db-data:
secrets:
  db-password:
    file: db/password.txt
```

The app is split into two parts:
1. The `server` app container with the python scraping functionality
2. The `db` container as a regular local postgres db service

The latter container features a persistent volume `db-data` in order to retain the data saved between container runs. The db container is not supposed to be stopped however, as it services the data for the data client lateron, e.g. PowerBI to process and analyse the data.

For the purpose of this local installation we proceed with default passwords.

### the market data web scraping app as python script


```python
#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup as bs
import cloudscraper
import json
import psycopg2

## subfunctions
def get_page(titleid):
    scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'firefox',
        'platform': 'windows',
        'mobile': False
        }
    )
    url = "https://barter.vg/i/" + str(titleid) + "/"
    print(url)
    r = scraper.get(url).text
    return r

def parse_page(r):
    import re

    soup = bs(r, 'html.parser')
    
    regex_t1 = r"Tradable <span>([0-9]+) \(([0-9]+)%\)</span></p>"
    regex_t2 = r"Tradable <span>([0-9]+)</span></p>"
    regex_w1 = r"Wishlist <span>([0-9]+) \(([0-9]+)%\)</span></p>"
    regex_w2 = r"Wishlist <span>([0-9]+)</span></p>"
    regex_l1 = r"Library <span>([0-9]+) \(([0-9]+)%\)</span></p>"
    regex_l2 = r"Library <span>([0-9]+)</span></p>"
    
    tradeables = soup.find("p", {"class": "listName bborder"})
    num_tradeables = re.findall(regex_t1, str(tradeables))
    if not num_tradeables:
        num_tradeables = re.findall(regex_t2, str(tradeables))
    print(tradeables)
    print(num_tradeables)
    print(num_tradeables[0][0])

    wishlisted = soup.find("p", {"class": "listName oborder"})
    num_wishlisted = re.findall(regex_w1, str(wishlisted))
    if not num_wishlisted:
        num_wishlisted = re.findall(regex_w2, str(wishlisted))
    print(wishlisted)
    print(num_wishlisted)
    print(num_wishlisted[0][0])

    inlibrary = soup.find("p", {"class": "listName gborder"})
    num_inlibrary = re.findall(regex_l1, str(inlibrary))
    if not num_inlibrary:
        num_inlibrary = re.findall(regex_l2, str(inlibrary))
    print(num_inlibrary[0][0])
    
    return num_tradeables[0][0], num_wishlisted[0][0], num_inlibrary[0][0]

# postgresql variant
# prep for docker


def postgres_db_check():
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host="python-docker-db-1",
            database="barter",
            user="postgres",
            password="mysecretpassword")

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        #print('PostgreSQL database version:')
        #cur.execute('SELECT version()')

        cur.execute('''CREATE TABLE IF NOT EXISTS gamedata
                                (
                                BARTERID       INT     NOT NULL,
                                DATE           TEXT    NOT NULL,
                                NUM_TRADEABLES INT,
                                NUM_WISHLISTED INT, 
                                NUM_INLIBRARY INT);'''
                   )
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
        conn.close()
    except:
        print('error')

def get_title_list():
        conn = psycopg2.connect(
        host="python-docker-db-1",
        database="barter",
        user="postgres",
        password="mysecretpassword",
        port=5432 # ex docker, internally 5432
        )

        # create a cursor
        cur = conn.cursor()

        cur.execute('''SELECT name, barterid FROM gamemeta;''' )
        
        records = cur.fetchall()

        #print("Print each row and it's columns values")
        #for row in records:
            #print("Id = ", row[0], )
            #print("Model = ", row[1],"\n")

        # close the communication with the PostgreSQL
        cur.close()
        conn.close()
        return records
        

def insert_scrape_into_postgres_table(NAME, BARTERID, DATE, NUM_TRADEABLES, NUM_WISHLISTED, NUM_INLIBRARY):
    try:
        conn = psycopg2.connect(
        host="python-docker-db-1",
        database="barter",
        user="postgres",
        password="mysecretpassword")

        cur = conn.cursor()
        print("Connected to Postgres")
        
        data_tuple = (BARTERID, DATE, NUM_TRADEABLES, NUM_WISHLISTED, NUM_INLIBRARY)
        
        cur.execute("""INSERT INTO gamedata(BARTERID, DATE, NUM_TRADEABLES, NUM_WISHLISTED, NUM_INLIBRARY) 
   VALUES (%s, %s, %s, %s, %s)""", data_tuple)
        
        conn.commit()
        print("Python Variables inserted successfully into table")
        
        cur.close()

    except psycopg2.Error as error:
        print("Failed to insert Python variables into table", error)
    finally:
        if conn:
            conn.close()
            print("The connection is closed")


# main function
from datetime import datetime
import time
import pandas as pd
import psycopg2
import random

def process_batch_postgres(save_in_db=False, return_df=False, debug=False):
    
    postgres_db_check()
            
    date = datetime.today().strftime('%Y-%m-%d')
    
    df = pd.DataFrame(columns=['NAME', 'BARTERID', 'DATE', 'NUM_TRADEABLES', 'NUM_WISHLISTED', 'NUM_INLIBRARY'])
        
    for title in get_title_list():
           
        print(title[0])
        r = get_page(title[1])
        if debug:
            print(r)
        
        num_tradeables, num_wishlisted, num_inlibrary = parse_page(r)
        
        if save_in_db:
            insert_scrape_into_postgres_table(title[0], title[1], date, num_tradeables, num_wishlisted, num_inlibrary)
        if return_df:
            df = df.append({'NAME':title[0], 'BARTERID': title[1], 'DATE':date, 
                            'NUM_TRADEABLES': int(num_tradeables), 'NUM_WISHLISTED':int(num_wishlisted), 'NUM_INLIBRARY':int(num_inlibrary)}, ignore_index=True)
        else:
            return_df = None
        
        time.sleep(random.randrange(10, 20)) #pacing the scrape 
    
    df.NUM_WISHLISTED = df.NUM_WISHLISTED.astype('int')
    df.NUM_TRADEABLES = df.NUM_TRADEABLES.astype('int')
    df['ratio'] = df.NUM_WISHLISTED / df.NUM_TRADEABLES
    return df


process_batch_postgres(save_in_db=True)
```

My App features `cloudscraper` and `BeautifulSoup` packages as part of the scraping functions `get_page` & `parse_page`. For the database part there is the `psycopg2` in service via functions `postgres_db_check`, `get_title_list`, `insert_scrape_into_postgres_table`. And lastly the main function that controls the application's runtime and calls all other functions.

To finally build this docker container I'll use 
`docker compose up --build` in a terminal prompt.

![My dockerized App in Docker Desktop](/blog/dockerize-market-data-scraping-app-w-python&postgresql_files/docker-desktop.png "My dockerized App in Docker Desktop")*My dockerized App in Docker Desktop*

The app in form of both containers is running. While the `db-1` container is supposed to be running permanently as a regular local postgres db service, the scraper in the `server-1` container finishes after completing a day's scrape.

![Accessing the containered postgres database with its Admin GUI locally](/blog/dockerize-market-data-scraping-app-w-python&postgresql_files/pgadmin-screen.png "Accessing the containered postgres database with its Admin GUI locally")*Accessing the containered postgres database with PGAdmin GUI locally*

Testing different SQL queries to assess the data and check if the app is producing error free data records.

![Modeled barter market data in a Power BI Dashboard analyzing data series over time](/blog/dockerize-market-data-scraping-app-w-python&postgresql_files/power-bi-dashboard.png "Modeled barter market data in a Power BI Dashboard analyzing data series over time")*Modeled barter market data in a Power BI Dashboard analyzing data series over time*

The data queries in this Power BI model are implemented as Direct Queries in order to avoid redundant data storage. This relates to the designated requirement of the database being available anytime and ensures always current data being displayed. The Dashboard's main purpose is to follow up on how demand vs. supply time series data evolves and their ratio to assess market situation at one glance.


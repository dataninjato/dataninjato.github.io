---
title: Auto ingest data to Snowflake with Snowpark Python API and RSA key based authentication
tags: [Data Engineering, Snowflake, Data Warehouse, data ingestion, Key based authentication, Python, SQL]
style: fill
color: info
description: Automate the ingestion of local or cloud based data files as part of our Data Engineering task to the cloud-based data warehouse Snowflake with help of Snowpark Python Libraries and API to process SnowSQL commands.
---
<style type="text/css">
 .markdown-body img:not(.emoji) {
    display: block;
    max-width: 1349px; !important
}
</style>

This blog is about that part of Data Engineering where we want to automate the ingestion data files locally or from S3 datalakes to the cloud-based data warehouse Snowflake with help of Snowpark Python Libraries and API to process SnowSQL commands.

For that we will utilize besides Snowflake itself, the Python package Snowpark and its API.

For the API authentication with Snowpark there are three options available:
1. regular basic authentication with login & password
2. Single Sign On (SSO) and 
3. Key based authentication.

I decide to use the latter Key based authentication as it is the preffered method anytime when SSH'ing into servers, saving time and preventing cli prompts that could hinder automation.

Let's get a fresh key created with openssl:


![RSA Key Generation](/blog/Auto-ingest-data-to-Snowflake-with-Snowpark-Python-API-and-RSA-key-based-authentication_files/rsa-keygen.png "RSA Key Generation")
*Generating an RSA key `rsa_key.p8` for authentication (in picture still password, later redone without)*

Having already setup an API users in my Snowflake instance [users/roles], it is time to add the public key to the user that has been assigned for running the Python script later.

![RSA Public Key User Asssignment](/blog/Auto-ingest-data-to-Snowflake-with-Snowpark-Python-API-and-RSA-key-based-authentication_files/pubkey-user-assignment-plus-key-check.png "RSA Public Key User Asssignment")
*Assigning the Public key to the designated user account plus verifying user account property "RSA_PUBLIC_KEY" for being set*

In order to prepare auto processing I have to set up a few things first, such as the Database itself, a named stage for incoming files (landing zone) and file formats that are required lateron.

![DB creation](/blog/Auto-ingest-data-to-Snowflake-with-Snowpark-Python-API-and-RSA-key-based-authentication_files/create-database.png "DB creation")
*Creating DB through GUI*

![named stage creation](/blog/Auto-ingest-data-to-Snowflake-with-Snowpark-Python-API-and-RSA-key-based-authentication_files/create-stage.png "named stage creation")
*Creating named stage through SQL WORKSHEET*

![file formats creation](/blog/Auto-ingest-data-to-Snowflake-with-Snowpark-Python-API-and-RSA-key-based-authentication_files/create-file-formats.png "file formats creation")
*Creating file formats through SQL WORKSHEET*

### auto ingestion process overview

1. upload designated files from local system into named stage (landing zone/bronze layer)
2. list files in stage and collect file names
3. Run infer schema
4. Identify column names & data-type
5. build DDL statements automatically for each file/table
6. save SQL DDL/COPY statements as SQL files
7. run DDL & COPY SQL commands


#### Step 1 - Upload designated files from local system into named stage (landing zone/bronze layer)

For file uploading I will use a first Python script that can automate the uploading of numerous files into the prepared named stage. It will utilize SnowSQL which is required anyway especially for large files and data chunks. With the PUT method files are uploaded sequentially. Our RSA keys are use for the key based authentication for the first time.

The final check confirms which files are uploaded and listed within the named stage 'BANK_STG'  along with its byte-size, md5 hash and last modified date.


```python
# loosely based on https://gist.github.com/vepetkov/a24b501e180cc5842b786ae680035239

import os
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import serialization

# Params
whName = "COMPUTE_WH"
dbName = "DB_SNOWPARK"
schemaName = "BANK_MARKETING"
stgName = "BANK_STG"
#filePath = "./bank-marketing-de/bank_marketing.csv"
fileArray = ["./bank-marketing-de/bank_marketing.csv",
             "./bank-marketing-de/campaign.csv",
             "./bank-marketing-de/client.csv",
             "./bank-marketing-de/economics.csv",
             ]

# rsa key based authentication
def snowpark_key_auth():
    with open("rsa_key.p8", "rb") as key:
        p_key= serialization.load_pem_private_key(
        key.read(),
        password=None,
        backend=default_backend()
        )

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())
 
    return pkb

# connect to snowpark
ctx = snowflake.connector.connect(
        account='yfnsuct-cc62406',
        user='API_SNOWPARK',
        private_key=snowpark_key_auth(),
        warehouse=whName,
        database=dbName,
        schema=schemaName
        )

cs = ctx.cursor()

# Step-1
for filePath in fileArray:
    # Upload each local file to the internal table stage
    uploadStmt = f'put file://{filePath} @{stgName} auto_compress=false;'
    cs.execute(f"use schema {schemaName};") # or set the Db & Scheme for the whole connection directly
    cs.execute(uploadStmt)

# Check the uploads
cs.execute(f"list @{stgName};")
uploadedFiles = cs.fetchall()
# check upload status
print(uploadedFiles)

cs.close()
ctx.close()
```

    [('bank_stg/bank_marketing.csv', 5094512, '6902dc210ce450be5c858bee3a18e55c', 'Wed, 25 Oct 2023 13:18:45 GMT'), ('bank_stg/campaign.csv', 1395408, '086c06bb40f01647ef58e095d40fb523', 'Wed, 25 Oct 2023 13:29:33 GMT'), ('bank_stg/client.csv', 2085824, '9d10bf6c799b7e8c8981913f5fcbd445', 'Wed, 25 Oct 2023 13:29:34 GMT'), ('bank_stg/economics.csv', 1229984, 'bafa17107ca5e51ba357a012dab01d43', 'Wed, 25 Oct 2023 13:29:36 GMT')]
    
#### Steps 2-7

```python
# based on DE Simplified course code & modified

import os
from datetime import datetime
import logging
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import serialization

def logging_setup():
    # create logger
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    logging.basicConfig(filename='debug.log', 
                        #encoding='utf-8', 
                        level=logging.DEBUG)
    return logger, ch

logger, ch = logging_setup()

# Params
whName = "COMPUTE_WH"
dbName = "DB_SNOWPARK"
schemaName = "BANK_MARKETING"
#stgName = "BANK_STG"

# separated to debug conda env packages
from snowflake.snowpark import Session

# rsa key based authentication
def snowpark_key_auth() -> Session:
    
    # load the private rsa key
    with open("rsa_key.p8", "rb") as key:
        p_key= serialization.load_pem_private_key(
            key.read(),
            password=None,
            backend=default_backend()
        )

    # private key text
    private_key_text = p_key.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption())
    
    connection_parameters = {
       "ACCOUNT":"yfnsuct-cc62406",
        "USER":"API_SNOWPARK",
        "private_key" : private_key_text
    }
    # creating snowflake session object
    return Session.builder.configs(connection_parameters).create()  

#Generate dynamic ddl statement based on column name and type list
def generate_ddl_statement(column_names, data_types, table_name):

    ddl_template = "CREATE TABLE IF NOT EXISTS {} (\n{});"

    columns = []
    for name, data_type in zip(column_names, data_types):
        column_definition = f"    {name} {data_type}"
        columns.append(column_definition)

    ddl_statement = ddl_template.format(table_name, ",\n".join(columns))

    print("Got the table creation ddl statement")

    return ddl_statement

#Generate dynamic copy statement based on table name    
def generate_copy_statement(table_name,stage_name,csv_file_path,file_format):
    copy_command = f"""
    COPY INTO {table_name}
    FROM @{stage_name}/{csv_file_path}
    FILE_FORMAT = (FORMAT_NAME = '{file_format}')
    """

    print(copy_command)
    return copy_command

# call the main function  
def main(snowpark_session,named_stg,copy_file_format,infer_schema_file_format):
    
    #set the context (database, schema and warehouse)
    snowpark_session.sql("use role sysadmin").collect()
    snowpark_session.sql(f"use database {dbName}").collect()
    snowpark_session.sql(f"use schema {schemaName}").collect()
    snowpark_session.sql(f"use warehouse {whName}").collect()

    # Step-2
    # fetch all files under the named stage location
    list_sql = "list @{}".format(named_stg)
    stg_files = snowpark_session.sql(list_sql).collect()

    # Step-3
    # Iterate through the stage file entires....
    for row in stg_files:
        row_value = row.as_dict()
        stg_file_path_value = row_value.get('name')

        #split the file path and extract names
        file_path, file_name = os.path.split(stg_file_path_value)
        stg_location = "@"+file_path

        # Dynamically create infer-schema SQL statement
        # by passing stage location + file name 
        infer_schema_sql = """ \
            SELECT *
            FROM TABLE(
                INFER_SCHEMA(
                LOCATION=>'{}/',
                files => '{}',
                FILE_FORMAT=>'{}'
            )
        )
        """.format(stg_location, file_name,infer_schema_file_format)

        print(infer_schema_sql)

        # Execute the infer schema SQL statement.
        inferred_schema_rows = snowpark_session.sql(infer_schema_sql).collect()

        col_name_lst = []
        col_data_type_lst = []

        # Step-4
        # Iterate through the list & extract column-name/type
        for row in inferred_schema_rows:
            row_value = row.as_dict()
            column_name = row_value.get('COLUMN_NAME')
            column_type = row_value.get('TYPE')
            #print(file_name.split('.')[0],column_name,column_type)
            col_name_lst.append(column_name)
            col_data_type_lst.append(column_type)
            
        print("---------------")
        print(col_name_lst)
        print(col_data_type_lst)

        table_name = file_name.split('.')[0]+"_tbl"
        # Step-5
        create_ddl_stmt = generate_ddl_statement(col_name_lst,col_data_type_lst, table_name.upper())
        copy_stmt = generate_copy_statement(table_name,named_stg,file_name,copy_file_format)
        
        # Step-6
        sql_file_path = table_name+".sql"
        with open(sql_file_path, "w") as sql_file:
            sql_file.write("-- Following statement is creating table\n")
            sql_file.write(create_ddl_stmt)
            sql_file.write("\n-- Following statement is executing copy command \n")
            sql_file.write(copy_stmt)

        # Execute the ddl statement + copy statement
        # Step-7
        snowpark_session.sql(create_ddl_stmt).collect()
        snowpark_session.sql(copy_stmt).collect()

#entry point  
if __name__ == '__main__':
    #capture start time
    utc_start_time = datetime.utcnow()

    #get snowpark session
    snowpark_session = snowpark_key_auth()

    #set tag before starting the process
    snowpark_session.sql("alter session set query_tag = 'Start Snowpark Automation'").collect()

    #stage name & fime format names.
    named_stg = "BANK_STG"
    whName = "COMPUTE_WH"
    dbName = "DB_SNOWPARK"
    schemaName = "BANK_MARKETING"   
    copy_file_format = "copy_csv_ff"
    infer_schema_file_format = "infer_schema_csv_ff"

    #calling main method
    main(snowpark_session,named_stg,copy_file_format,infer_schema_file_format) 

    #record end time
    utc_end_time = datetime.utcnow()
    
    #set the query tag
    snowpark_session.sql("alter session set query_tag = 'End Snowpark Automation'").collect()

    #print overall execution time..
    print("Execution Time", utc_end_time-utc_start_time)
```

                 SELECT *
                FROM TABLE(
                    INFER_SCHEMA(
                    LOCATION=>'@bank_stg/',
                    files => 'bank_marketing.csv',
                    FILE_FORMAT=>'infer_schema_csv_ff'
                )
            )
            
    ---------------
    ['client_id', 'age', 'job', 'marital', 'education', 'credit_default', 'housing', 'loan', 'contact', 'month', 'day', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']
    ['NUMBER(5, 0)', 'NUMBER(2, 0)', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'NUMBER(2, 0)', 'NUMBER(4, 0)', 'NUMBER(2, 0)', 'NUMBER(3, 0)', 'NUMBER(1, 0)', 'TEXT', 'NUMBER(2, 1)', 'NUMBER(5, 3)', 'NUMBER(3, 1)', 'NUMBER(4, 3)', 'NUMBER(5, 1)', 'BOOLEAN']
    Got the table creation ddl statement
    
        COPY INTO bank_marketing_tbl
        FROM @BANK_STG/bank_marketing.csv
        FILE_FORMAT = (FORMAT_NAME = 'copy_csv_ff')
        
                 SELECT *
                FROM TABLE(
                    INFER_SCHEMA(
                    LOCATION=>'@bank_stg/',
                    files => 'campaign.csv',
                    FILE_FORMAT=>'infer_schema_csv_ff'
                )
            )
            
    ---------------
    ['client_id', 'number_contacts', 'contact_duration', 'pdays', 'previous_campaign_contacts', 'previous_outcome', 'campaign_outcome', 'campaign_id', 'last_contact_date']
    ['NUMBER(5, 0)', 'NUMBER(2, 0)', 'NUMBER(4, 0)', 'NUMBER(3, 0)', 'NUMBER(1, 0)', 'NUMBER(2, 1)', 'NUMBER(1, 0)', 'NUMBER(1, 0)', 'DATE']
    Got the table creation ddl statement
    
        COPY INTO campaign_tbl
        FROM @BANK_STG/campaign.csv
        FILE_FORMAT = (FORMAT_NAME = 'copy_csv_ff')
        
                 SELECT *
                FROM TABLE(
                    INFER_SCHEMA(
                    LOCATION=>'@bank_stg/',
                    files => 'client.csv',
                    FILE_FORMAT=>'infer_schema_csv_ff'
                )
            )
            
    ---------------
    ['id', 'age', 'job', 'marital', 'education', 'credit_default', 'housing', 'loan']
    ['NUMBER(5, 0)', 'NUMBER(2, 0)', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT']
    Got the table creation ddl statement
    
        COPY INTO client_tbl
        FROM @BANK_STG/client.csv
        FILE_FORMAT = (FORMAT_NAME = 'copy_csv_ff')
        
                 SELECT *
                FROM TABLE(
                    INFER_SCHEMA(
                    LOCATION=>'@bank_stg/',
                    files => 'economics.csv',
                    FILE_FORMAT=>'infer_schema_csv_ff'
                )
            )
            
    ---------------
    ['client_id', 'emp_var_rate', 'cons_price_idx', 'euribor_three_months', 'number_employed']
    ['NUMBER(5, 0)', 'NUMBER(2, 1)', 'NUMBER(5, 3)', 'NUMBER(4, 3)', 'NUMBER(5, 1)']
    Got the table creation ddl statement
    
        COPY INTO economics_tbl
        FROM @BANK_STG/economics.csv
        FILE_FORMAT = (FORMAT_NAME = 'copy_csv_ff')
        
    Execution Time 0:00:06.682118
    

Contents of auto created SQL statement file in step 6 of the script `bank_marketing_tbl.sql`.
With this file I am able to review the inferred data types and eventually change and if required:
``` SQL
-- Following statement is creating table
CREATE TABLE IF NOT EXISTS BANK_MARKETING_TBL (
    client_id NUMBER(5, 0),
    age NUMBER(2, 0),
    job TEXT,
    marital TEXT,
    education TEXT,
    credit_default TEXT,
    housing TEXT,
    loan TEXT,
    contact TEXT,
    month TEXT,
    day NUMBER(2, 0),
    duration NUMBER(4, 0),
    campaign NUMBER(2, 0),
    pdays NUMBER(3, 0),
    previous NUMBER(1, 0),
    poutcome TEXT,
    emp_var_rate NUMBER(2, 1),
    cons_price_idx NUMBER(5, 3),
    cons_conf_idx NUMBER(3, 1),
    euribor3m NUMBER(4, 3),
    nr_employed NUMBER(5, 1),
    y BOOLEAN);
-- Following statement is executing copy command 

    COPY INTO bank_marketing_tbl
    FROM @BANK_STG/bank_marketing.csv
    FILE_FORMAT = (FORMAT_NAME = 'copy_csv_ff')
```    


![Snowflake GUI auto ingested tables](/blog/Auto-ingest-data-to-Snowflake-with-Snowpark-Python-API-and-RSA-key-based-authentication_files/automatically-ingested-files-with-python-snowpark.png "Snowflake GUI auto ingested tables")
*Snowflake GUI showing the auto ingested table BANK_MARKETING_TBL*

### Conclusion 


These scripts provide a quick start ingesting new data files and SQL statement handling, while being flexible and adjustable to one requirements.
While being basic and universal these scripts could be a basis for more sophisticated automations, error detection & reporting etc pp.

I enjoyed Snowflake & Snowpark Python packages very much, as they were working reliable and smoothly. The codebase has apparently matured and no bugs have been noticed.

Next coming up I plan on progressing to silver and gold layers and start consuming the data, preferably NoSQL Data. This is  Snowflake's strengths to support attribute specific queries of semi-structured data such as JSON/XML. 


```python

```

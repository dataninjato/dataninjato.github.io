---
title: Realtime streaming data analytics with PyFlink, Apache Kafka, Elasticsearch & Kibana
tags: [PyFlink, Apache Kafka, Elasticsearch, Kibana, Python, Data Analytics, Streaming Data]
style: fill
color: danger
description: Analyse and monitor business data & payment amounts in ten retail branches realtime with a Kafka/Elasticsearch/Kibana setup.
---

Last weekend I spun up a few docker containers for an Apache Kafka, Elasticsearch, Kibana Stack Setup to play around the capabilities of the Python PyFlink package using the official Apache pyflink-walkthrough. 
As an open-source stream processing framework, PyFlink extends Flink's powerful stream processing features to Python developers like myself. This allows me to construct scalable, high-throughput, and fault-tolerant stream processing applications using Python.

We will first
1. Save and Extract data from a Kafka source.
2. Transform data with a User Defined Function (UDF).
3. Use basic aggregation on the sourced data.
4. Store the outcomes in Elasticsearch.
5. Present realtime data through Kibana visualizations.

#### Use Cases for Streaming with Kafka

PyFlink and Kafka offer powerful capabilities for real-time stream processing e.g. in retail and ecommerce. With PyFlink's ability to analyze data streams in Python and Kafka's efficient data handling, businesses can address various use cases:

- **Real-Time Inventory Management:** Optimize inventory levels and prevent stockouts.
- **Dynamic Pricing:** Adjust prices based on real-time market conditions and demand/low stock situations (see above). Detect pricing errors.
- **Personalized Marketing:** Deliver tailored marketing campaigns and product recommendations.
- **Fraud Detection:** Identify and prevent fraudulent transactions instantly.
- **Supply Chain Optimization:** Improve forecasting accuracy and reduce logistics costs.
- **Customer Experience Monitoring:** Gain insights into satisfaction levels and address concerns promptly.

By leveraging PyFlink and Kafka, retailers and ecommerce businesses can enhance efficiency, drive revenue, and improve customer satisfaction.

We build images and compose up with this ```docker-compose.yml```:


```python
################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

version: '2.1'
services:
  jobmanager:
    build: .
    image: pyflink/pyflink:1.16.0-scala_2.12
    volumes:
      - .:/opt/pyflink-walkthrough
    hostname: "jobmanager"
    expose:
      - "6123"
    ports:
      - "8081:8081"
    command: jobmanager
    environment:
      - JOB_MANAGER_RPC_ADDRESS=jobmanager
  taskmanager:
    image: pyflink/pyflink:1.16.0-scala_2.12
    volumes:
    - .:/opt/pyflink-walkthrough
    expose:
      - "6121"
      - "6122"
    depends_on:
      - jobmanager
    command: taskmanager
    links:
      - jobmanager:jobmanager
    environment:
      - JOB_MANAGER_RPC_ADDRESS=jobmanager
  zookeeper:
    image: wurstmeister/zookeeper:3.4.6
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
    ports:
      - "2181:2181"
  kafka:
    image: wurstmeister/kafka:2.13-2.8.1
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper
    environment:
      HOSTNAME_COMMAND: "route -n | awk '/UG[ \t]/{print $$2}'"
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_CREATE_TOPICS: "payment_msg:1:1"
  generator:
    build: generator
    image: generator:1.0
    depends_on:
      - kafka
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.8.0
    environment:
      - cluster.name=docker-cluster
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - discovery.type=single-node
    ports:
      - "9200:9200"
      - "9300:9300"
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
  kibana:
    image: docker.elastic.co/kibana/kibana:7.8.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
  load-kibana-dashboard:
    build: ./kibana
    command: ['/bin/bash', '-c', 'cat /tmp/load/load_ndjson.sh | tr -d "\r" | bash']
    depends_on:
      - kibana
```

lets produce a data stream with ```generate_source_data.py```
and adapt it to a retailer that wants to monitor payment amounts of its 10 branches:


```python
################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import random
import time, calendar
from random import randint
from kafka import KafkaProducer
from kafka import errors 
from json import dumps
from time import sleep

def write_data(producer):
    data_cnt = 20000
    order_id = calendar.timegm(time.gmtime())
    max_price = 1000
    topic = "payment_msg"

    for i in range(data_cnt):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        rd = random.random()
        order_id += 1
        pay_amount = max_price * rd
        pay_platform = 0 if random.random() < 0.9 else 1
        branch_id = randint(0, 9)
        cur_data = {"createTime": ts, "orderId": order_id, "payAmount": pay_amount, "payPlatform": pay_platform, "branchId": branch_id}
        producer.send(topic, value=cur_data)
        sleep(1)

def create_producer():
    print("Connecting to Kafka brokers")
    for i in range(0, 6):
        try:
            producer = KafkaProducer(bootstrap_servers=['kafka:9092'],
                            value_serializer=lambda x: dumps(x).encode('utf-8'))
            print("Connected to Kafka")
            return producer
        except errors.NoBrokersAvailable:
            print("Waiting for brokers to become available")
            sleep(10)

    raise RuntimeError("Failed to connect to brokers within 60 seconds")

if __name__ == '__main__':
    producer = create_producer()
    write_data(producer)

```

this produces this data stream saved in Kafka in topic ```payment_msg``` monitored with 

```sudo docker-compose exec kafka kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic payment_msg```



![data-generator-running-in-terminal](/blog/data-generator-running-in-terminal.png "data-generator-running-in-terminal")

A source table for Kafka is defined, as well as a data sink in Elasticsearch

In between the transactional data will be processed with PyFlink utilizing a user defined function that maps the branchId to its very branch name. In a second step the payment amounts will aggregate over all transactions and grouped by branches.

```payment_msg_processing.py```:


```python
################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.table import StreamTableEnvironment, DataTypes, EnvironmentSettings
from pyflink.table.expressions import call, col
from pyflink.table.udf import udf


branches = [f'Branch_{i}' for i in range(0, 10)]

@udf(input_types=[DataTypes.STRING()], result_type=DataTypes.STRING())
def branch_id_to_name(id):
    return branches[id]


def log_processing():
    env = StreamExecutionEnvironment.get_execution_environment()
    t_env = StreamTableEnvironment.create(stream_execution_environment=env)
    t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)

    create_kafka_source_ddl = """
            CREATE TABLE payment_msg(
                createTime VARCHAR,
                orderId BIGINT,
                payAmount DOUBLE,
                payPlatform INT,
                branchId INT
            ) WITH (
              'connector' = 'kafka',
              'topic' = 'payment_msg',
              'properties.bootstrap.servers' = 'kafka:9092',
              'properties.group.id' = 'test_3',
              'scan.startup.mode' = 'latest-offset',
              'format' = 'json'
            )
            """

    create_es_sink_ddl = """
            CREATE TABLE es_sink(
                branch VARCHAR PRIMARY KEY,
                pay_amount DOUBLE
            ) with (
                'connector' = 'elasticsearch-7',
                'hosts' = 'http://elasticsearch:9200',
                'index' = 'platform_pay_amount_1',
                'document-id.key-delimiter' = '$',
                'sink.bulk-flush.max-size' = '42mb',
                'sink.bulk-flush.max-actions' = '32',
                'sink.bulk-flush.interval' = '1000',
                'sink.bulk-flush.backoff.delay' = '1000',
                'format' = 'json'
            )
    """

    t_env.execute_sql(create_kafka_source_ddl)
    t_env.execute_sql(create_es_sink_ddl)
    t_env.register_function('branch_id_to_name', branch_id_to_name)

    t_env.from_path("payment_msg") \
        .select(call('branch_id_to_name', col('branchId')).alias("branch"), col('payAmount')) \
        .group_by(col('branch')) \
        .select(col('branch'), call('sum', col('payAmount').alias("pay_amount"))) \
        .execute_insert("es_sink")


if __name__ == '__main__':
    log_processing()
```

```docker-compose exec jobmanager ./bin/flink run -py /opt/pyflink-walkthrough/payment_msg_proccessing.py -d```

submits the Flink Job and can be tracked in the Apache Flink Dashboard

![Apache-flink-dashboard-running-job-status](/blog/Apache-flink-dashboard-running-job-status.png "Apache-flink-dashboard-running-job-status")

The processed and stored data in Elasticsearch can be visualized immediately using Kibana, illustrating which branches are busy and which are less active.

![Kibana-dashboard-realtime-total-payment-amounts](/blog/Kibana-dashboard-realtime-total-payment-amounts.png "Kibana-dashboard-realtime-total-payment-amounts")

Due to my Power BI PL-300 Exam coming up this month I will have to cut short and 
note to myself a to do list:

- Adjust indexes of the Elasticsearch sink
- Aggregate data by Time windows such as tumbling or sliding windows.
- Define multiple sinks
- Test the powerful pandas integration & built-in transformational functions

tbc
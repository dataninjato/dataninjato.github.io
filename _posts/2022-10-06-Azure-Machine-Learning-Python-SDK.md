---
title: Azure Machine Learning Python SDK Deployment of Airline Delay Data Processing & Classification Modelling Pipeline
tags: [Python, Azure, Azure Machine Learning, Azure ML Python SDK, Pipeline, Machine Learning, Classification, SHAP]
style: border
color: primary
comments: true
description: Complete Pipeline workflow including registering data set on Azure and uploading files into the datastore, converting former notebook to a Python script for the Preprocessing Pipeline step to be run and secondly the compact ML script to complete the Pipeline before running it. Finally having a quick look which features are of high importance in predicting delays.
---


### Implementing Data Processing & Classification Modelling Pipeline

At first I demonstrate how to apply the Azure ML Python SDK to implement a complete Data Processing & Classification Modelling Pipeline Datastores from a regular notebook that could be even deployed as endpoint as a Webservice with REST API and scheduled to be updated with new data over night. But as there is no use use case for anybody to consume this model, I will instead point the focus on explaining the model's output through the data features after the machine learning process has been completed on Azure ML Cloud.


```python
import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()

print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
```

    Ready to use Azure ML 1.44.0 to work with azure-ml-ws-flight-delay
    


```python
from azureml.core import Dataset
from azureml.data.datapath import DataPath

default_ds = ws.get_default_datastore()

dataset_name = 'airline_delay_ds'
dataset_description = 'Airline Delay and Cancellation Data, 2009 - 2018. Flight information of US domestic flights'
dataset_path = 'airline-delay-data/'
source_path = 'data'

if dataset_name not in ws.datasets:
    Dataset.File.upload_directory(src_dir=source_path,
                              target=DataPath(default_ds, dataset_path)
                              )

    #Create a tabular dataset from the path on the datastore (this may take a short while)
    tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, dataset_path + '*.csv'))

    # Register the tabular dataset
    try:
        tab_data_set = tab_data_set.register(workspace=ws, 
                                name=dataset_name,
                                description=dataset_description,
                                tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    print('Dataset already registered.')
```

    Dataset already registered.
    


```python
import os
# Create a folder for the pipeline step files
pipeline_folder = '_pipeline'
os.makedirs(pipeline_folder, exist_ok=True)

print(pipeline_folder)
```

    _pipeline
    

Write the first script reading and preprocessing from our airline delay dataset including the normalization of numeric features for equal scales.

The script includes a argument named **--prepped-data**, which references the folder where the processed data is to be saved.


```python
%%writefile _pipeline/prep_airline_delay.py
# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from azureml.data.datapath import DataPath
from azureml.core import Workspace

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

# load the data (passed as an input dataset)
print("Loading Data...")
df1 = run.input_datasets['raw_data'].to_pandas_dataframe()

# import modified derivate script of ipynb and execute it
exec(open('Data_Cleaning_Preprocessing_all.py').read())

# Log raw row count
row_count = (len(dfm2))
run.log('raw_rows', row_count)

# stratified sample
dfm2 = dfm2.groupby('FLIGHT_STATUS', group_keys=False).apply(lambda x: x.sample(frac=0.5))

print(dfm2.shape)
print(dfm2.head())

col_names = ['DEP_TIME',
 'DEP_DELAY',
 'TAXI_OUT',
 'TAXI_IN',
 'ARR_TIME',
 'CRS_ELAPSED_TIME',
 'ACTUAL_ELAPSED_TIME',
 'AIR_TIME',
 'DISTANCE']

for column in dfm2[col_names]:
    print(column)
    dfm2[column] = dfm2[column].astype('float32')
    dfm2[column] = StandardScaler().fit_transform(dfm2[[column]])

print("Current Path: " + os.getcwd())

print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'df_for_modeling.parquet.gzip')

dfm2.to_parquet(save_path, compression='gzip')

# save processed data as new dataset

"""from azureml.core import Workspace, Dataset

subscription_id = 'cc376adb-00a6-44ff-9100-212ed0161c43'
resource_group = 'default_resource_group'
workspace_name = 'azure-ml-ws-flight-delay'

ws = Workspace(subscription_id, resource_group, workspace_name)
default_ds = ws.get_default_datastore()
dataset_name = 'processed_airline_delay_ds'
dataset_description = 'Proccessed Airline Delay and Cancellation Data 2018'
dataset_path = 'processed-airline-delay-data/'


if dataset_name not in ws.datasets:
    
    try:
        #Create a tabular dataset from the path on the datastore (this may take a short while)
        tab_data_set = Dataset.Tabular.register_pandas_dataframe(dataframe=dfm2, target=dataset_path, name=dataset_name)
        print('Dataset created.')

    except Exception as ex:
       print(ex)

    # Register the tabular dataset
    try:
        tab_data_set = tab_data_set.register(workspace=ws, 
                                name=dataset_name,
                                #description=dataset_description,
                                #tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    print('Dataset already registered.')
"""
# End the run
run.complete()
```

    Overwriting _pipeline/prep_airline_delay.py
    

Now you can create the script for the second step, which will train a model. The script includes a argument named **--training-data**, which references the location where the prepared data was saved by the previous step.


```python
%%writefile _pipeline/train_airline_delay.py

# Import libraries
from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os
np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)

import datetime, warnings, scipy
warnings.filterwarnings("ignore")

from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
#from tensorflow import keras

import gc
import psutil
from azureml.interpret import ExplanationClient
from azureml.core.run import Run
from interpret.ext.blackbox import TabularExplainer
from mlxtend.plotting import plot_confusion_matrix

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
args = parser.parse_args()
training_data = args.training_data

# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print("Loading Data...")
print("Current Path: " + os.getcwd())
"""try:
    df = pd.read_parquet('df_for_modeling.parquet.gzip')
except:
    df = pd.read_parquet('../df_for_modeling.parquet.gzip')"""

print("Loading Data...")
file_path = os.path.join(training_data,'df_for_modeling.parquet.gzip')
df = pd.read_parquet(file_path)

# Separate features and labels
y = df['FLIGHT_STATUS']
X = df.drop(['FLIGHT_STATUS'], axis=1)

print(psutil.Process().memory_info().rss / (1024 * 1024))

print('before any gc', psutil.Process().memory_info().rss / (1024 * 1024))

# clear df from memory
del df
gc.collect()
print('after 1st gc', psutil.Process().memory_info().rss / (1024 * 1024))

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# clear df from memory
del X
del y
gc.collect()

print('after 2nd gc', psutil.Process().memory_info().rss / (1024 * 1024))

print(X_train.shape)

X_train.info()

# Train adecision tree model
#print('Training a decision tree model...')
#model = DecisionTreeClassifier().fit(X_train, y_train)

print('x shape' , X_train.shape[1])

mlp = tf.keras.models.Sequential()
mlp.add(tf.keras.layers.Dense(50, activation='tanh', input_shape=(X_train.shape[1],)))
mlp.add(tf.keras.layers.Dense(30, activation='tanh'))
mlp.add(tf.keras.layers.Dense(15, activation='tanh'))
mlp.add(tf.keras.layers.Dense(5, activation='relu'))
mlp.add(tf.keras.layers.Dense(1, activation='sigmoid'))
mlp.summary()

# calculate accuracy
#y_hat = model.predict(X_test)
#acc = np.average(y_hat == y_test)
#print('Accuracy:', acc)
#run.log('Accuracy', np.float(acc))

mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results = mlp.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

y_pred = mlp.predict(X_test)
y_pred = (y_pred > 0.5) # threshold 

#run.log('confusion matrix', confusion_matrix(y_test, y_pred))
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred), show_absolute=True, show_normed=True, colorbar=True)
plt.title('Confusion Matrix - Airline Delay Classification', fontsize=14)
run.log_image(name = "Confusion Matrix", plot = fig)
plt.show()

run.log('classification_report', classification_report(y_test, y_pred))
run.log('accuracy', metrics.accuracy_score(y_test, y_pred))
run.log('precision', precision_score(y_test, y_pred))
run.log('recall', recall_score(y_test, y_pred))
run.log('f1_score', f1_score(y_test, y_pred))

acc = metrics.accuracy_score(y_test, y_pred)

# calculate AUC
y_scores = mlp.predict_proba(X_test)
auc = roc_auc_score(y_test,y_pred)
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

fig = plt.figure(figsize=(10, 6))
plt.plot(results.history['val_loss'])
plt.plot(results.history['loss'])
plt.legend(['val_loss', 'loss'])
plt.title('LOSS', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
run.log_image(name = "Loss", plot = fig)
plt.show()

fig = plt.figure(figsize=(10, 6))
plt.plot(results.history['val_accuracy'])
plt.plot(results.history['accuracy'])
plt.legend(['val_accuracy', 'accuracy'])
plt.title('ACCURACY', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
run.log_image(name = "Accuracy", plot = fig)
plt.show()

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'model')
#joblib.dump(value=mlp, filename=model_file)

mlp.save(model_file)
print('tf model saved', str(model_file))
#except:
#    print('tf save model did not work')

# model explanation
client = ExplanationClient.from_run(run)

"""# explain the model and upload
print('Tabular Explainer')
explainer = TabularExplainer(mlp, X_train, features=X_train.columns, #classes=['not delayed', 'delayed']
                            )
global_explanation = explainer.explain_global(X_test)
#client.upload_model_explanation(global_explanation, top_k=30, comment='global explanation: Only top 30 features')
client.upload_model_explanation(global_explanation, comment='global explanation: all features')
"""

# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'airline_delay_model',
               tags={'Training context':'Pipeline'},
               properties={'AUC': np.float(auc), 'Accuracy': np.float(acc)})
run.complete()
```

    Overwriting _pipeline/train_airline_delay.py
    


```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "Standard-E4ds-v4"

try:
    # Check for existing compute target
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_E2s_v3', max_nodes=3)
        pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
```

    Found existing cluster, use it.
    


```python
%%writefile $pipeline_folder/pipeline_env.yml
name: pipeline_env
dependencies:
- python=3.8
- numpy
- joblib
- scikit-learn
- Keras==2.3.1
- tensorflow
- ipykernel
- matplotlib
- seaborn
- statsmodels
- scipy
- pandas
- pip
- pip:
  - azureml-defaults
  - azureml-interpret
  - mlxtend
  - pyarrow
```

    Overwriting _pipeline/pipeline_env.yml
    

Now that you have a Conda configuration file, you can create an environment and use it in the run configuration for the pipeline.


```python
from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration

# Create a Python environment for the experiment (from a .yml file)
pipeline_env = Environment.from_conda_specification("pipeline_env", pipeline_folder + "/pipeline_env.yml")

# Register the environment 
pipeline_env.register(workspace=ws)
registered_env = Environment.get(ws, 'pipeline_env')

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = pipeline_cluster

# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

print ("Run configuration created.")
```

    Run configuration created.
    


```python
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep

# Get the training dataset
train_ds = ws.datasets.get(dataset_name)

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
prepped_data = OutputFileDatasetConfig(destination=(default_ds, 'airline-delay-pipeline/outputdataset'))
#prepped_data = OutputFileDatasetConfig("prepped_data")

# Step 1, Run the data prep script
prep_step = PythonScriptStep(name = "Prepare Data",
                                source_directory = pipeline_folder,
                                script_name = "prep_airline_delay.py",
                                arguments = ['--input-data', train_ds.as_named_input('raw_data'),
                                             '--prepped-data', prepped_data],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

# Step 2, run the training script
train_step = PythonScriptStep(name = "Train and Register Model",
                                source_directory = pipeline_folder,
                                script_name = "train_airline_delay.py",
                                arguments = ['--training-data', prepped_data.as_input()],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

print("Pipeline steps defined")
```

    Pipeline steps defined
    


```python
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.widgets import RunDetails

# Construct the pipeline
pipeline_steps = [prep_step, train_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name = 'airline_delay-pipeline')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")
RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)
```

    Pipeline is built.
    Created step Prepare Data [4a72cbf2][3ce48b4b-bff1-4bf8-a8f3-ca06443c3be9], (This step will run and generate new outputs)
    Created step Train and Register Model [f226a213][c408023b-6638-4778-9aa4-e70b24d40e33], (This step will run and generate new outputs)
    Submitted PipelineRun 9ba5c47b-6a01-481d-a364-0f950b4f7206
    Link to Azure Machine Learning Portal: https://ml.azure.com/runs/9ba5c47b-6a01-481d-a364-0f950b4f7206?wsid=/subscriptions/cc376adb-00a6-44ff-9100-212ed0161c43/resourcegroups/default_resource_group/workspaces/azure-ml-ws-flight-delay&tid=722595d7-167c-4a9a-b847-dcf4de414b79
    Pipeline submitted for execution.
    


    _PipelineWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO', …




    PipelineRunId: 9ba5c47b-6a01-481d-a364-0f950b4f7206
    Link to Azure Machine Learning Portal: https://ml.azure.com/runs/9ba5c47b-6a01-481d-a364-0f950b4f7206?wsid=/subscriptions/cc376adb-00a6-44ff-9100-212ed0161c43/resourcegroups/default_resource_group/workspaces/azure-ml-ws-flight-delay&tid=722595d7-167c-4a9a-b847-dcf4de414b79
    PipelineRun Status: Running
    
    
    StepRunId: 73947786-1972-4aa0-9464-3864ed95284a
    Link to Azure Machine Learning Portal: https://ml.azure.com/runs/73947786-1972-4aa0-9464-3864ed95284a?wsid=/subscriptions/cc376adb-00a6-44ff-9100-212ed0161c43/resourcegroups/default_resource_group/workspaces/azure-ml-ws-flight-delay&tid=722595d7-167c-4a9a-b847-dcf4de414b79
    StepRun( Prepare Data ) Status: Running
    
    StepRun(Prepare Data) Execution Summary
    ========================================
    StepRun( Prepare Data ) Status: Finished
    {'runId': '73947786-1972-4aa0-9464-3864ed95284a', 'target': 'Standard-E4ds-v4', 'status': 'Completed', 'startTimeUtc': '2022-10-05T18:03:07.340605Z', 'endTimeUtc': '2022-10-05T18:06:42.580367Z', 'services': {}, 'properties': {'ContentSnapshotId': 'a50368b7-7cdc-44b9-a5af-1f0c8dd6b350', 'StepType': 'PythonScriptStep', 'ComputeTargetType': 'AmlCompute', 'azureml.moduleid': '3ce48b4b-bff1-4bf8-a8f3-ca06443c3be9', 'azureml.moduleName': 'Prepare Data', 'azureml.runsource': 'azureml.StepRun', 'azureml.nodeid': '4a72cbf2', 'azureml.pipelinerunid': '9ba5c47b-6a01-481d-a364-0f950b4f7206', 'azureml.pipeline': '9ba5c47b-6a01-481d-a364-0f950b4f7206', 'azureml.pipelineComponent': 'masterescloud', '_azureml.ComputeTargetType': 'amlctrain', 'ProcessInfoFile': 'azureml-logs/process_info.json', 'ProcessStatusFile': 'azureml-logs/process_status.json'}, 'inputDatasets': [{'dataset': {'id': 'e62077d8-168a-4e5f-9039-4d4700ff1837'}, 'consumptionDetails': {'type': 'RunInput', 'inputName': 'raw_data', 'mechanism': 'Direct'}}], 'outputDatasets': [{'identifier': {'savedId': '0394a826-2f1e-47ea-8b53-7e56620ca9ee'}, 'outputType': 'RunOutput', 'outputDetails': {'outputName': 'output_a4a84fad'}, 'dataset': {
      "source": [
        "('workspaceblobstore', 'airline-delay-pipeline/outputdataset')"
      ],
      "definition": [
        "GetDatastoreFiles"
      ],
      "registration": {
        "id": "0394a826-2f1e-47ea-8b53-7e56620ca9ee",
        "name": null,
        "version": null,
        "workspace": "Workspace.create(name='azure-ml-ws-flight-delay', subscription_id='cc376adb-00a6-44ff-9100-212ed0161c43', resource_group='default_resource_group')"
      }
    }}], 'runDefinition': {'script': 'prep_airline_delay.py', 'command': '', 'useAbsolutePath': False, 'arguments': ['--input-data', 'DatasetConsumptionConfig:raw_data', '--prepped-data', 'DatasetOutputConfig:output_a4a84fad'], 'sourceDirectoryDataStore': None, 'framework': 'Python', 'communicator': 'None', 'target': 'Standard-E4ds-v4', 'dataReferences': {}, 'data': {'raw_data': {'dataLocation': {'dataset': {'id': 'e62077d8-168a-4e5f-9039-4d4700ff1837', 'name': None, 'version': '1'}, 'dataPath': None, 'uri': None, 'type': None}, 'mechanism': 'Direct', 'environmentVariableName': 'raw_data', 'pathOnCompute': None, 'overwrite': False, 'options': None}}, 'outputData': {'output_a4a84fad': {'outputLocation': {'dataset': None, 'dataPath': {'datastoreName': 'workspaceblobstore', 'relativePath': 'airline-delay-pipeline/outputdataset'}, 'uri': None, 'type': None}, 'mechanism': 'Mount', 'additionalOptions': {'pathOnCompute': None, 'registrationOptions': {'name': None, 'description': None, 'tags': None, 'properties': {'azureml.pipelineRunId': '9ba5c47b-6a01-481d-a364-0f950b4f7206', 'azureml.pipelineRun.moduleNodeId': '4a72cbf2', 'azureml.pipelineRun.outputPortName': 'output_a4a84fad'}, 'datasetRegistrationOptions': {'additionalTransformation': None}}, 'uploadOptions': {'overwrite': False, 'sourceGlobs': {'globPatterns': None}}, 'mountOptions': None}, 'environmentVariableName': None}}, 'datacaches': [], 'jobName': None, 'maxRunDurationSeconds': None, 'nodeCount': 1, 'instanceTypes': [], 'priority': None, 'credentialPassthrough': False, 'identity': None, 'environment': {'name': 'pipeline_env', 'version': '10', 'assetId': 'azureml://locations/germanywestcentral/workspaces/be7c6676-a074-4c76-8a0a-3342397f971c/environments/pipeline_env/versions/10', 'autoRebuild': True, 'python': {'interpreterPath': 'python', 'userManagedDependencies': False, 'condaDependencies': {'name': 'pipeline_env', 'dependencies': ['python=3.8', 'numpy', 'joblib', 'scikit-learn', 'Keras==2.3.1', 'tensorflow', 'ipykernel', 'matplotlib', 'seaborn', 'statsmodels', 'scipy', 'pandas', 'pip', {'pip': ['azureml-defaults', 'azureml-interpret', 'mlxtend', 'pyarrow']}]}, 'baseCondaEnvironment': None}, 'environmentVariables': {'EXAMPLE_ENV_VAR': 'EXAMPLE_VALUE'}, 'docker': {'baseImage': 'mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20220708.v1', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'baseDockerfile': None, 'baseImageRegistry': {'address': None, 'username': None, 'password': None}, 'enabled': False, 'arguments': []}, 'spark': {'repositories': [], 'packages': [], 'precachePackages': True}, 'inferencingStackVersion': None}, 'history': {'outputCollection': True, 'directoriesToWatch': ['logs'], 'enableMLflowTracking': True, 'snapshotProject': True}, 'spark': {'configuration': {'spark.app.name': 'Azure ML Experiment', 'spark.yarn.maxAppAttempts': '1'}}, 'parallelTask': {'maxRetriesPerWorker': 0, 'workerCountPerNode': 1, 'terminalExitCodes': None, 'configuration': {}}, 'amlCompute': {'name': None, 'vmSize': None, 'retainCluster': False, 'clusterMaxNodeCount': 1}, 'aiSuperComputer': {'instanceType': 'D2', 'imageVersion': 'pytorch-1.7.0', 'location': None, 'aiSuperComputerStorageData': None, 'interactive': False, 'scalePolicy': None, 'virtualClusterArmId': None, 'tensorboardLogDirectory': None, 'sshPublicKey': None, 'sshPublicKeys': None, 'enableAzmlInt': True, 'priority': 'Medium', 'slaTier': 'Standard', 'userAlias': None}, 'kubernetesCompute': {'instanceType': None}, 'tensorflow': {'workerCount': 1, 'parameterServerCount': 1}, 'mpi': {'processCountPerNode': 1}, 'pyTorch': {'communicationBackend': 'nccl', 'processCount': None}, 'hdi': {'yarnDeployMode': 'Cluster'}, 'containerInstance': {'region': None, 'cpuCores': 2.0, 'memoryGb': 3.5}, 'exposedPorts': None, 'docker': {'useDocker': False, 'sharedVolumes': True, 'shmSize': '2g', 'arguments': []}, 'cmk8sCompute': {'configuration': {}}, 'commandReturnCodeConfig': {'returnCode': 'Zero', 'successfulReturnCodes': []}, 'environmentVariables': {}, 'applicationEndpoints': {}, 'parameters': []}, 'logFiles': {'logs/azureml/dataprep/0/backgroundProcess.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/logs/azureml/dataprep/0/backgroundProcess.log?sv=2019-07-07&sr=b&sig=YqHUzDdnwmNM7K7l7Q70Gh8E9eFenJpQdSTG%2BJl6KiU%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A53%3A18Z&se=2022-10-06T02%3A03%3A18Z&sp=r', 'logs/azureml/dataprep/0/backgroundProcess_Telemetry.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/logs/azureml/dataprep/0/backgroundProcess_Telemetry.log?sv=2019-07-07&sr=b&sig=7DPkwEC9XwxRDuwWEWpMqeQAGE0eaS2L6RH6zaacrM8%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A53%3A18Z&se=2022-10-06T02%3A03%3A18Z&sp=r', 'logs/azureml/dataprep/0/rslex.log.2022-10-05-18': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/logs/azureml/dataprep/0/rslex.log.2022-10-05-18?sv=2019-07-07&sr=b&sig=S7ogkKHnd8VU1GtOKiW4BRUwAb8YpIlGsUWRWeZjNoU%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A53%3A18Z&se=2022-10-06T02%3A03%3A18Z&sp=r', 'logs/azureml/executionlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/logs/azureml/executionlogs.txt?sv=2019-07-07&sr=b&sig=8lySVHxXDRsc2%2F8R9YIpDVMz%2BUJo8sbod3FyO9uznC8%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A53%3A18Z&se=2022-10-06T02%3A03%3A18Z&sp=r', 'logs/azureml/stderrlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/logs/azureml/stderrlogs.txt?sv=2019-07-07&sr=b&sig=IOE0yFKVd9zvgW76%2FMc8NR25BDY31IUA8T1yZuSb6YU%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A53%3A18Z&se=2022-10-06T02%3A03%3A18Z&sp=r', 'logs/azureml/stdoutlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/logs/azureml/stdoutlogs.txt?sv=2019-07-07&sr=b&sig=%2BIW%2BF0FCxrVmNwVin8Wk%2FPtgPwwNIHU%2Bk69c%2Bp%2BLNgs%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A53%3A18Z&se=2022-10-06T02%3A03%3A18Z&sp=r', 'user_logs/std_log.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/user_logs/std_log.txt?sv=2019-07-07&sr=b&sig=9jnCW708rRVooQUbe5dJFWnbeYHmyrR5YrAQKrc%2Bhp4%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/cs_capability/cs-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/cs_capability/cs-capability.log?sv=2019-07-07&sr=b&sig=nNEmlQXjoz1g91293C3deODydfMMh3JB%2FBaY%2FoF%2Bc34%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/data_capability/data-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/data_capability/data-capability.log?sv=2019-07-07&sr=b&sig=6Mt4Z3msYem2kruUsYUFR94c%2B2L%2BEBrPX2yTBQPn3mU%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/data_capability/rslex.log.2022-10-05-18': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/data_capability/rslex.log.2022-10-05-18?sv=2019-07-07&sr=b&sig=tuV9AV%2BTALjjbvJkNKokvMXpygSQuZxrlA1adNtir7o%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/hosttools_capability/hosttools-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/hosttools_capability/hosttools-capability.log?sv=2019-07-07&sr=b&sig=pjSmOBOD%2BZJO137h6zzNuSGpPsKPNWLLYnGIZD1qZtU%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/lifecycler/execution-wrapper.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/lifecycler/execution-wrapper.log?sv=2019-07-07&sr=b&sig=eeC37VLS51MofJcdIHWmye42%2F3d5IEdhztk1z2RMZKs%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/lifecycler/lifecycler.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/lifecycler/lifecycler.log?sv=2019-07-07&sr=b&sig=lsNY8re5kZoIMa%2FCbEmWOE1xMOo4Sv1Wt%2BQkk1PbCSk%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/metrics_capability/metrics-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/metrics_capability/metrics-capability.log?sv=2019-07-07&sr=b&sig=aTzKHBydqYXwTM4Lp5BGsrfUvO75v4xUFM3wgITmRX4%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r', 'system_logs/snapshot_capability/snapshot-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.73947786-1972-4aa0-9464-3864ed95284a/system_logs/snapshot_capability/snapshot-capability.log?sv=2019-07-07&sr=b&sig=5pFo5FCfbLwmUC1ZiSICYbuMGAWiPOeyDBwURUVODhw%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T17%3A56%3A45Z&se=2022-10-06T02%3A06%3A45Z&sp=r'}, 'submittedBy': 'Sean Flight'}
    
    
    
    
    StepRunId: 4250a1ce-6edc-4ea9-b139-7eae4a49650f
    Link to Azure Machine Learning Portal: https://ml.azure.com/runs/4250a1ce-6edc-4ea9-b139-7eae4a49650f?wsid=/subscriptions/cc376adb-00a6-44ff-9100-212ed0161c43/resourcegroups/default_resource_group/workspaces/azure-ml-ws-flight-delay&tid=722595d7-167c-4a9a-b847-dcf4de414b79
    StepRun( Train and Register Model ) Status: Running
    
    StepRun(Train and Register Model) Execution Summary
    ====================================================
    StepRun( Train and Register Model ) Status: Finished
    {'runId': '4250a1ce-6edc-4ea9-b139-7eae4a49650f', 'target': 'Standard-E4ds-v4', 'status': 'Completed', 'startTimeUtc': '2022-10-05T18:07:17.608113Z', 'endTimeUtc': '2022-10-05T19:10:44.434993Z', 'services': {}, 'properties': {'ContentSnapshotId': 'a50368b7-7cdc-44b9-a5af-1f0c8dd6b350', 'StepType': 'PythonScriptStep', 'ComputeTargetType': 'AmlCompute', 'azureml.moduleid': 'c408023b-6638-4778-9aa4-e70b24d40e33', 'azureml.moduleName': 'Train and Register Model', 'azureml.runsource': 'azureml.StepRun', 'azureml.nodeid': 'f226a213', 'azureml.pipelinerunid': '9ba5c47b-6a01-481d-a364-0f950b4f7206', 'azureml.pipeline': '9ba5c47b-6a01-481d-a364-0f950b4f7206', 'azureml.pipelineComponent': 'masterescloud', '_azureml.ComputeTargetType': 'amlctrain', 'ProcessInfoFile': 'azureml-logs/process_info.json', 'ProcessStatusFile': 'azureml-logs/process_status.json'}, 'inputDatasets': [{'dataset': {'id': '0394a826-2f1e-47ea-8b53-7e56620ca9ee'}, 'consumptionDetails': {'type': 'RunInput', 'inputName': 'input_75af8948', 'mechanism': 'Mount'}}], 'outputDatasets': [], 'runDefinition': {'script': 'train_airline_delay.py', 'command': '', 'useAbsolutePath': False, 'arguments': ['--training-data', 'DatasetConsumptionConfig:input_75af8948'], 'sourceDirectoryDataStore': None, 'framework': 'Python', 'communicator': 'None', 'target': 'Standard-E4ds-v4', 'dataReferences': {}, 'data': {'input_75af8948': {'dataLocation': {'dataset': {'id': '0394a826-2f1e-47ea-8b53-7e56620ca9ee', 'name': None, 'version': None}, 'dataPath': None, 'uri': None, 'type': None}, 'mechanism': 'Mount', 'environmentVariableName': 'input_75af8948', 'pathOnCompute': None, 'overwrite': False, 'options': None}}, 'outputData': {}, 'datacaches': [], 'jobName': None, 'maxRunDurationSeconds': None, 'nodeCount': 1, 'instanceTypes': [], 'priority': None, 'credentialPassthrough': False, 'identity': None, 'environment': {'name': 'pipeline_env', 'version': '10', 'assetId': 'azureml://locations/germanywestcentral/workspaces/be7c6676-a074-4c76-8a0a-3342397f971c/environments/pipeline_env/versions/10', 'autoRebuild': True, 'python': {'interpreterPath': 'python', 'userManagedDependencies': False, 'condaDependencies': {'name': 'pipeline_env', 'dependencies': ['python=3.8', 'numpy', 'joblib', 'scikit-learn', 'Keras==2.3.1', 'tensorflow', 'ipykernel', 'matplotlib', 'seaborn', 'statsmodels', 'scipy', 'pandas', 'pip', {'pip': ['azureml-defaults', 'azureml-interpret', 'mlxtend', 'pyarrow']}]}, 'baseCondaEnvironment': None}, 'environmentVariables': {'EXAMPLE_ENV_VAR': 'EXAMPLE_VALUE'}, 'docker': {'baseImage': 'mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20220708.v1', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'baseDockerfile': None, 'baseImageRegistry': {'address': None, 'username': None, 'password': None}, 'enabled': False, 'arguments': []}, 'spark': {'repositories': [], 'packages': [], 'precachePackages': True}, 'inferencingStackVersion': None}, 'history': {'outputCollection': True, 'directoriesToWatch': ['logs'], 'enableMLflowTracking': True, 'snapshotProject': True}, 'spark': {'configuration': {'spark.app.name': 'Azure ML Experiment', 'spark.yarn.maxAppAttempts': '1'}}, 'parallelTask': {'maxRetriesPerWorker': 0, 'workerCountPerNode': 1, 'terminalExitCodes': None, 'configuration': {}}, 'amlCompute': {'name': None, 'vmSize': None, 'retainCluster': False, 'clusterMaxNodeCount': 1}, 'aiSuperComputer': {'instanceType': 'D2', 'imageVersion': 'pytorch-1.7.0', 'location': None, 'aiSuperComputerStorageData': None, 'interactive': False, 'scalePolicy': None, 'virtualClusterArmId': None, 'tensorboardLogDirectory': None, 'sshPublicKey': None, 'sshPublicKeys': None, 'enableAzmlInt': True, 'priority': 'Medium', 'slaTier': 'Standard', 'userAlias': None}, 'kubernetesCompute': {'instanceType': None}, 'tensorflow': {'workerCount': 1, 'parameterServerCount': 1}, 'mpi': {'processCountPerNode': 1}, 'pyTorch': {'communicationBackend': 'nccl', 'processCount': None}, 'hdi': {'yarnDeployMode': 'Cluster'}, 'containerInstance': {'region': None, 'cpuCores': 2.0, 'memoryGb': 3.5}, 'exposedPorts': None, 'docker': {'useDocker': False, 'sharedVolumes': True, 'shmSize': '2g', 'arguments': []}, 'cmk8sCompute': {'configuration': {}}, 'commandReturnCodeConfig': {'returnCode': 'Zero', 'successfulReturnCodes': []}, 'environmentVariables': {}, 'applicationEndpoints': {}, 'parameters': []}, 'logFiles': {'logs/azureml/executionlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/logs/azureml/executionlogs.txt?sv=2019-07-07&sr=b&sig=7P1qOx0fm6tIc3kNfe1IFsARCvSgcHFWPIj%2FthgLibI%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T18%3A56%3A59Z&se=2022-10-06T03%3A06%3A59Z&sp=r', 'logs/azureml/stderrlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/logs/azureml/stderrlogs.txt?sv=2019-07-07&sr=b&sig=NNXno1tmGqFybJGHxUPMYim%2Fm688Nsd9ck5F46lx%2FXs%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T18%3A56%3A59Z&se=2022-10-06T03%3A06%3A59Z&sp=r', 'logs/azureml/stdoutlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/logs/azureml/stdoutlogs.txt?sv=2019-07-07&sr=b&sig=lRg%2Bwh%2FxLfvKwfA94Dph6S5rxKWkQmwspK%2FBn9mHIF4%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T18%3A56%3A59Z&se=2022-10-06T03%3A06%3A59Z&sp=r', 'user_logs/std_log.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/user_logs/std_log.txt?sv=2019-07-07&sr=b&sig=g%2FB7Diis5ar9kXUqqltV0huQO9CNKjS0lkOyd40YZAk%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A29Z&ske=2022-10-06T23%3A46%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/cs_capability/cs-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/cs_capability/cs-capability.log?sv=2019-07-07&sr=b&sig=XIigkWaavlYmLk69yLcCUq0SGN4ezUc05dN7AtXJ5OQ%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/data_capability/data-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/data_capability/data-capability.log?sv=2019-07-07&sr=b&sig=fJcHS1nqlejVFOko1YL2KKx2KL%2BX7odGcFqArNsd%2F2U%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/data_capability/rslex.log.2022-10-05-18': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/data_capability/rslex.log.2022-10-05-18?sv=2019-07-07&sr=b&sig=8D3pIr9jI%2FNzUu9aITWWR8GW8dY2Q5cyZaP%2F2i8ERnY%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/data_capability/rslex.log.2022-10-05-19': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/data_capability/rslex.log.2022-10-05-19?sv=2019-07-07&sr=b&sig=jrqPlnyFwtQR8P8eQZfUB5QApUChHN56oUM95HH5A9o%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/hosttools_capability/hosttools-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/hosttools_capability/hosttools-capability.log?sv=2019-07-07&sr=b&sig=QtfSbcjV1Zo7xlZeh8fIJOeNn9dWmcExnqPNZPxlbx8%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/lifecycler/execution-wrapper.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/lifecycler/execution-wrapper.log?sv=2019-07-07&sr=b&sig=2R%2Bixq1nLg79UAzszeZw75Xf%2FOoSwXxVi4axAwlDbxI%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/lifecycler/lifecycler.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/lifecycler/lifecycler.log?sv=2019-07-07&sr=b&sig=tiXuHyuWYPImw%2FNLoZ8xU8l6BYLQLr3m7Sb2Fz3GxpM%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/metrics_capability/metrics-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/metrics_capability/metrics-capability.log?sv=2019-07-07&sr=b&sig=NMrdoChHsUSOz%2Bh%2Fypp06phortM2jLROJKSS46UB0w4%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r', 'system_logs/snapshot_capability/snapshot-capability.log': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.4250a1ce-6edc-4ea9-b139-7eae4a49650f/system_logs/snapshot_capability/snapshot-capability.log?sv=2019-07-07&sr=b&sig=rju2fWMce%2BsWV0fMQfSqFBDNwnAcjp1Fhi%2BQIRPWjUE%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A36%3A27Z&ske=2022-10-06T23%3A46%3A27Z&sks=b&skv=2019-07-07&st=2022-10-05T19%3A00%3A46Z&se=2022-10-06T03%3A10%3A46Z&sp=r'}, 'submittedBy': 'Sean Flight'}
    
    
    
    PipelineRun Execution Summary
    ==============================
    PipelineRun Status: Finished
    {'runId': '9ba5c47b-6a01-481d-a364-0f950b4f7206', 'status': 'Completed', 'startTimeUtc': '2022-10-05T17:58:51.854239Z', 'endTimeUtc': '2022-10-05T19:10:45.803538Z', 'services': {}, 'properties': {'azureml.runsource': 'azureml.PipelineRun', 'runSource': 'SDK', 'runType': 'SDK', 'azureml.parameters': '{}', 'azureml.continue_on_step_failure': 'False', 'azureml.continue_on_failed_optional_input': 'True', 'azureml.pipelineComponent': 'pipelinerun'}, 'inputDatasets': [], 'outputDatasets': [], 'logFiles': {'logs/azureml/executionlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.9ba5c47b-6a01-481d-a364-0f950b4f7206/logs/azureml/executionlogs.txt?sv=2019-07-07&sr=b&sig=LU0zJwNmPeezV7lF%2BvClOgqQG7gX7YD1TqDu%2FfKyHP0%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T18%3A58%3A00Z&se=2022-10-06T03%3A08%3A00Z&sp=r', 'logs/azureml/stderrlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.9ba5c47b-6a01-481d-a364-0f950b4f7206/logs/azureml/stderrlogs.txt?sv=2019-07-07&sr=b&sig=QdMQBKhMMLF3kbuUTcdFXucWmmif9LT6WsiRqWM%2F57s%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T18%3A58%3A00Z&se=2022-10-06T03%3A08%3A00Z&sp=r', 'logs/azureml/stdoutlogs.txt': 'https://azuremlwsfligh8207488120.blob.core.windows.net/azureml/ExperimentRun/dcid.9ba5c47b-6a01-481d-a364-0f950b4f7206/logs/azureml/stdoutlogs.txt?sv=2019-07-07&sr=b&sig=BBpBOrFnS3SunL25%2BJCPcGQgzszlY0KeABVx%2Bx7%2BU1g%3D&skoid=3b36aaa3-38b6-425b-9ec8-dd8762ceabd9&sktid=722595d7-167c-4a9a-b847-dcf4de414b79&skt=2022-10-05T15%3A54%3A29Z&ske=2022-10-07T00%3A04%3A29Z&sks=b&skv=2019-07-07&st=2022-10-05T18%3A58%3A00Z&se=2022-10-06T03%3A08%3A00Z&sp=r'}, 'submittedBy': 'Sean Flight'}
    
    




    'Finished'




```python
for run in pipeline_run.get_children():
    print(run.name, ':')
    metrics = run.get_metrics()
    for metric_name in metrics:
        print('\t',metric_name, ":", metrics[metric_name])
```

    Train and Register Model :
    	 Confusion Matrix : ![png](/blog/airline-delay-shap-explainer-files/Confusion Matrix_1664997001.png)
    	 classification_report :               precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    552126
               1       1.00      1.00      1.00    332425
    
        accuracy                           1.00    884551
       macro avg       1.00      1.00      1.00    884551
    weighted avg       1.00      1.00      1.00    884551
    
    	 accuracy : 0.998930530856898
    	 precision : 0.9977954449206631
    	 recall : 0.9993622621643979
    	 f1_score : 0.9985782389400215
    	 AUC : 0.999016427735499
    	 Loss : ![png](/blog/airline-delay-shap-explainer-files/Loss_1664997037.png)
    	 Accuracy : ![png](/blog/airline-delay-shap-explainer-files/Accuracy_1664997037.png)
    Prepare Data :
    	 raw_rows : 7076405
    


```python
from azureml.core import Model

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')
```

    airline_delay_model version: 10
    	 Training context : Pipeline
    	 AUC : 0.999016427735499
    	 Accuracy : 0.998930530856898
    
    
    airline_delay_model version: 8
    	 Training context : Pipeline
    	 AUC : 0.9989681954777851
    	 Accuracy : 0.9990526266998737
    
    
    airline_delay_model version: 7
    	 Training context : Pipeline
    	 AUC : 0.8937972057441084
    	 Accuracy : 0.9062146892655367
		 
		 
![png](/blog/airline-delay-shap-explainer-files/strong_clock_zjxtw17d-Microsoft-Azure-Machine-Learning-Studio.png)

    
### Explain the black box Neural Network model

Besides the Azure Pipeline implementation let us have a quick look at which features have predictive value towards the delay label.

This is just a starting point, a beachhead for an in depth analysis of the model that aims at determining the factors that correlate with airline delays or may be even causing them.


```python
# import required modules

import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning) 

# load model
model = load_model('airline_delay_model10_')
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 50)                42350     
    _________________________________________________________________
    dense_1 (Dense)              (None, 30)                1530      
    _________________________________________________________________
    dense_2 (Dense)              (None, 15)                465       
    _________________________________________________________________
    dense_3 (Dense)              (None, 5)                 80        
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 6         
    =================================================================
    Total params: 44,431
    Trainable params: 44,431
    Non-trainable params: 0
    _________________________________________________________________
    


```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet('model10-files/df_for_modeling.parquet.gzip')
```


```python
y = df['FLIGHT_STATUS']
X = df.drop(['FLIGHT_STATUS'], axis=1)
# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, random_state=0)
```


```python
X_train.shape
```




    (3537849, 846)




```python
X_test.shape
```




    (354, 846)




```python
y_preds_proba = model.predict(X_test)
#y_preds_proba
```


```python
y_preds = (model.predict(X_test)>0.5).astype(int)
#y_preds
```


```python
import sklearn
import shap
from sklearn.model_selection import train_test_split

print("SHAP Version : {}".format(shap.__version__))

# print the JS visualization code to the notebook
shap.initjs()
```

    SHAP Version : 0.39.0
    

![png](/blog/airline-delay-shap-explainer-files/shap-force-plot-instance0a.png)


The same force plot transformed with logit to probabilities:


```python
i=0

shap.force_plot(explainer.expected_value, shap_values[0][i], X_test.iloc[i], link="logit")
```





![png](/blog/airline-delay-shap-explainer-files/shap-force-plot-instance0b.png)



```python
values = shap_values[0]
base_values = [explainer.expected_value[0]]*len(shap_values[0])

tmp = shap.Explanation(values = np.array(values, dtype=np.float32),
                       base_values = np.array(base_values, dtype=np.float32),
                       data=np.array(X_test),
                       feature_names=X_test.columns)

shap.plots.waterfall(tmp[i])
```


    
![png](/blog/airline-delay-shap-explainer-files/airline%20delay%20blog2-for-real%20interpret%20shap%20lime%20explainer_20_0.png)
    


##### Instance 8

Instance 8 is a flight that has been clearly delayed before departure. This increases the probability for a arrival delay tremendously. Forces indicating the arrival to delay are colored red and factors rather indicating being punctual are drawn in blue.


```python
i=8

shap.force_plot(explainer.expected_value, shap_values[0][i], X_test.iloc[i], link="logit")
```





![png](/blog/airline-delay-shap-explainer-files/shap-force-plot-instance8.png)



```python
values = shap_values[0]
base_values = [explainer.expected_value[0]]*len(shap_values[0])

tmp = shap.Explanation(values = np.array(values, dtype=np.float32),
                       base_values = np.array(base_values, dtype=np.float32),
                       data=np.array(X_test),
                       feature_names=X_test.columns)

shap.plots.waterfall(tmp[i])
```


    
![png](/blog/airline-delay-shap-explainer-files/airline%20delay%20blog2-for-real%20interpret%20shap%20lime%20explainer_23_0.png)
    



```python
shap.force_plot(explainer.expected_value, shap_values[0][2], X_test.iloc[2], link="logit")
```







##### Instance 3

is an example for a closer call between factors both pushing into opposite directions red and blue approx. equally strong but blue prevails in this model predicting a `0 not delayed` label.


```python
i=3

shap.force_plot(explainer.expected_value, shap_values[0][i], X_test.iloc[i], link="logit")
```




![png](/blog/airline-delay-shap-explainer-files/shap-force-plot-instance3.png)





```python
values = shap_values[0]
base_values = [explainer.expected_value[0]]*len(shap_values[0])

tmp = shap.Explanation(values = np.array(values, dtype=np.float32),
                       base_values = np.array(base_values, dtype=np.float32),
                       data=np.array(X_test),
                       feature_names=X_test.columns)

shap.plots.waterfall(tmp[i])
```


    
![png](/blog/airline-delay-shap-explainer-files/airline%20delay%20blog2-for-real%20interpret%20shap%20lime%20explainer_27_0.png)
    

    
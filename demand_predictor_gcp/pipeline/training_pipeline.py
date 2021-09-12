
import os
import kfp
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret
from kfp.components import func_to_container_op
from helper_components import evaluate_model
from helper_components import retrieve_best_run
import kfp.dsl as dsl
import kfp.gcp as gcp
import json


# Defaults and environment settings
BASE_IMAGE = os.getenv('BASE_IMAGE')
TRAINER_IMAGE = os.getenv('TRAINER_IMAGE')
RUNTIME_VERSION = os.getenv('RUNTIME_VERSION')
PYTHON_VERSION = os.getenv('PYTHON_VERSION')
BUCKET = os.getenv('BUCKET')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')
USE_KFP_SA = os.getenv('USE_KFP_SA')

TRAINING_DATA_PATH = BUCKET + '/data/dataset.csv'
COMPONENT_URL_SEARCH_PREFIX = 'https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/'

# Create component factories
component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

HYPERTUNE_SETTINGS = """
{
    "hyperparameters":  {
        "goal": "MAXIMIZE",
        "maxTrials": 4,
        "maxParallelTrials": 2,
        "hyperparameterMetricTag": "mean_absolute_error)",
        "enableTrialEarlyStopping": True,
        "params": [
            {
                "parameterName": "epochs",
                "type": "DISCRETE",
                "discreteValues": [500, 1000]
            },
            {
                "parameterName": "lr",
                "type": "DOUBLE",
                "minValue": 0.0001,
                "maxValue": 0.001,
                "scaleType": "UNIT_LINEAR_SCALE"
            }
        ]
    }
"""


# Create component factories
component_store = kfp.components.ComponentStore(
    local_search_paths=None,
    url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

# Load BigQuery and AI Platform Training op
# bigquery_query_op = component_store.load_component('bigquery/query')
mlengine_train_op = component_store.load_component('ml_engine/train')
mlengine_deploy_op = component_store.load_component('ml_engine/deploy')

# dsl pipeline definition
@dsl.pipeline(
    name='Spanish Demand forecast Continuous Training',
    description='Pipeline to create training/validation on AI Platform Training Job'
)
def pipeline(project_id,
             gcs_root,
             model_id,
             version_id,
             replace_existing_version,
             region='us-central1',
             hypertune_settings=HYPERTUNE_SETTINGS):

    # These are the output directories where our models will be saved
    output_dir = project_id + '/models/pipeline'
    
    # Tune hyperparameters
    tune_args = [
        '--training_dataset_path', TRAINING_DATA_PATH,
        '--hptune', 'True']

    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir/hypertune',
                                kfp.dsl.RUN_ID_PLACEHOLDER)

    hypertune = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=tune_args,
        training_input=hypertune_settings)
    
    # Retrieve the best trial
    get_best_trial = retrieve_best_run_op(project_id, hypertune.outputs['job_id'])

    # Train the model on a combined training and validation datasets
    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir',
                                kfp.dsl.RUN_ID_PLACEHOLDER)
    
    train_args = [
        '--training_dataset_path', TRAINING_DATA_PATH,
        '--output_dir', output_dir,
        '--window_size', '30',
        '--batch_size', '16', 
        get_best_trial.outputs['lr'], '--lr',
        get_best_trial.outputs['epochs'], '--epochs'
        '--hptune', 'False'
    ]

    train_model = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=train_args).set_display_name('Tensorflow Model Training')
    
    deploy_model = mlengine_deploy_op(
        model_uri=train_model.outputs['job_dir'],
        project_id=project_id,
        model_id=model_id,
        version_id=version_id,
        runtime_version=RUNTIME_VERSION,
        python_version=PYTHON_VERSION,
        replace_existing_version=replace_existing_version)

    # Configure the pipeline to run using the service account defined
    # in the user-gcp-sa k8s secret
    if USE_KFP_SA == 'True':
        kfp.dsl.get_pipeline_conf().add_op_transformer(
              use_gcp_secret('user-gcp-sa'))

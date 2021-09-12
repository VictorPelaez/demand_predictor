
import os
import kfp
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret
import kfp.components as comp
import kfp.dsl as dsl
import kfp.gcp as gcp
import json

# We will use environment vars to set the trainer image names and bucket name
TF_TRAINER_IMAGE = os.getenv('TF_TRAINER_IMAGE')
BUCKET = os.getenv('BUCKET')

# Paths to export the training/validation data from bigquery
#TRAINING_OUTPUT_PATH = BUCKET + '/data/training.csv'

TRAINING_OUTPUT_PATH = "gs://qwiklabs-gcp-04-699809b7c735-kubeflowpipelines-default/data/dataset.csv"

COMPONENT_URL_SEARCH_PREFIX = 'https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/'

# Create component factories
component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

# Load BigQuery and AI Platform Training op
# bigquery_query_op = component_store.load_component('bigquery/query')
mlengine_train_op = component_store.load_component('ml_engine/train')


@dsl.pipeline(
    name='Spanish Demand forecast Continuous Training',
    description='Pipeline to create training/validation on AI Platform Training Job'
)
def pipeline(
    project_id,
    region='us-central1'
):

    # These are the output directories where our models will be saved
    tf_output_dir = BUCKET + '/models/tf'

    # Training arguments to be passed to the TF Trainer
    tf_args = [
        '--training_dataset_path', TRAINING_OUTPUT_PATH,
        '--output_dir', tf_output_dir,
        '--window_size', '30',
        '--batch_size', '16', 
        '--epochs', '1000',
        '--lr', '1e-3'
    ]
    
    # AI Platform Training Job     
    train_tf = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TF_TRAINER_IMAGE,
        args=tf_args).set_display_name('Tensorflow Model - AI Platform Training')

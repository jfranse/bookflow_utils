import yaml
import time
from PIL import Image

from myst_nb import glue
import mlflow
import mlflow.entities.run

class BookflowConfig():
    def __init__(self, config_file):
        self.tracking_uri = None
        self.experiment_name = None
        self.config_file = config_file
        self.load_yaml_config(config_file)

    def load_yaml_config(self, config_file):
        with open(config_file, 'r') as stream:
            config_dict = yaml.safe_load(stream)
        for k,v in config_dict.items():
            setattr(self, k, v)


class BookflowHelper():
    def __init__(self, config: BookflowConfig):
        self.config = config
        self.tracking_uri = config.tracking_uri
        self.experiment_name = config.experiment_name
        self.experiment_id = self.get_experiment_id()

    def get_experiment_id(self, experiment_name=None):
        return get_experiment_id(experiment_name if experiment_name else self.experiment_name)

    def get_run_latest(self):
        return get_run_latest(self.experiment_id)



def set_tracking_uri(cls: BookflowConfig):
    mlflow.set_tracking_uri(cls.tracking_uri)
    print(f'MLFlow tracking URI set to {cls.tracking_uri}')

def get_run_latest(experiment_id):
    return mlflow.get_run(mlflow.search_runs(experiment_ids=[experiment_id], max_results=1).loc[0].run_id)

def get_experiment_id(experiment_name):
    return mlflow.get_experiment_by_name(experiment_name).experiment_id

def glue_image_by_uri(glue_name, image_uri):
    im = Image.open(image_uri)
    glue(glue_name, im, display=False)

def glue_model_reference_metadata(run: mlflow.entities.run.Run, glue_name=None):
    internal_meta_data = dict(
        commit_hash=run.data.tags['mlflow.source.git.commit'],
        run_id=run.info.run_id,
        run_end_time_local=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.info.end_time / 1000.)),
    )

    if glue_name is None: glue_name = 'model_ref_metedata'

    glue(glue_name, internal_meta_data)
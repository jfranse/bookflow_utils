import yaml
import time
from PIL import Image
import pandas as pd

from typing import List, Optional

from myst_nb import glue
import mlflow
import mlflow.entities.run

from bookflow_utils.mlflow_tools import get_latest_run, get_experiment_id, get_params_as_df

class BookflowConfig():
    def __init__(self, config_file):
        self.tracking_uri = None
        self.experiment_name = None
        self.default_valid_run_tag = None
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
        self.default_valid_run_tag = config.default_valid_run_tag

        self.set_tracking_uri()
        self.experiment_id = self.get_experiment_id()

    def get_experiment_id(self, experiment_name=None):
        return get_experiment_id(experiment_name if experiment_name else self.experiment_name)

    def get_latest_run(self, tags=None, custom_query=None):
        if tags is None: tags = self.default_valid_run_tag
        return get_latest_run(self.experiment_id, tags=tags, custom_query=custom_query)

    def set_tracking_uri(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        print(f'MLFlow tracking URI set to {self.tracking_uri}')

    def glue_image_by_uri(self, glue_name, image_uri):
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

    def extract_journal_entry_data(self, run: mlflow.entities.run.Run):
        header = dict(
            datetime=time.strftime('%Y-%m-%d %H:%M', time.localtime(run.info.end_time / 1000.)),
            run_name=run.data.tags['mlflow.runName'],
            description=run.data.tags['mlflow.note.content'],
        )
        parameters_table = (pd.DataFrame(run.data.params.items(), columns=['Parameter', 'Value']).
                            set_index('Parameter', drop=True))
        metrics_table = (pd.DataFrame(run.data.metrics.items(), columns=['Metric', 'Value']).
                            set_index('Metric', drop=True))
        return header, parameters_table, metrics_table

    def get_params_as_df(self,
                         run: mlflow.entities.run.Run,
                         drop: Optional[List[str]] = None
                         ) -> pd.DataFrame:
        return get_params_as_df(run, drop=drop)

def glue_image_by_uri(glue_name, image_uri):
    im = Image.open(image_uri)
    glue(glue_name, im, display=False)
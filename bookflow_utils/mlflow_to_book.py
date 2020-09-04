import yaml
import time
from PIL import Image
import pandas as pd
from pathlib import Path

from typing import List, Optional, Dict, Tuple, Union

from myst_nb import glue
import mlflow
import mlflow.entities.run

from bookflow_utils.mlflow_tools import get_latest_run, get_experiment_id, get_params_as_df, get_metrics_as_df

class BookflowConfig():

    config_template = dict(
        tracking_uri='/path/to/mlruns',
        experiment_name='experiment_name',
        target_tag={'main_tag': 'True'}
    )

    def __init__(self, config_file=None, tracking_uri=None, experiment_name=None, target_tag=None):
        """Set the configuration. Config from config file will take precedence over keyword arguments.

        Args:
            config_file: path to config file
            tracking_uri: path to where your MLFlor runs are stored
            experiment_name: the MLFlow experiment name you want to load from
            target_tag: dictionary of {'tagname':'value'} that will be passed to the mlflow search to select runs.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.target_tag = target_tag
        self.config_file = config_file
        if self.config_file: self.load_yaml_config(config_file)


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
        self.target_tag = config.target_tag

        self.set_tracking_uri()
        self.experiment_id = self.get_experiment_id()

        self.latest_run = self.get_latest_run()

    def get_experiment_id(self, experiment_name=None):
        """Retrun MLFlow experiment id. Defaults to the configured experiment name, but you can supply a different one."""
        return get_experiment_id(experiment_name if experiment_name else self.experiment_name)

    def get_latest_run(self, tags='default', custom_query=None) -> mlflow.entities.run.Run:
        """Return the latest MLFlow Run from the configured experiment.
        By default only selects from runs with the configured target tag, but this behaviour can be superseded..

        Args:
            tags: by default only runs with the configured tag:value are selected. Replace with your own {'tag':'value'} or None
            custom_query: supply a custom query string to be appended to the search if you want.

        Returns:
            MLFlor Run object for the most recent run.
        """
        if tags == 'default': tags = self.target_tag
        return get_latest_run(self.experiment_id, tags=tags, custom_query=custom_query)

    def set_tracking_uri(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        print(f'MLFlow tracking URI set to {self.tracking_uri}')

    def glue_param(self, glue_ref: str, param: str, run: mlflow.entities.run.Run = None):
        """Glue the parameter 'param' to the glue reference name 'glue_ref'.
        By default for the configured latest run, but you are free to supply any Run object."""
        if run is None: run = self.latest_run
        glue(glue_ref, run.data.params[param], display=False)

    def glue_metric(self, glue_ref: str, metric: str, run: mlflow.entities.run.Run = None):
        """Glue the parameter 'metric' to the glue reference name 'glue_ref'
        By default for the configured latest run, but you are free to supply any Run object."""
        if run is None: run = self.latest_run
        glue(glue_ref, run.data.metrics[metric], display=False)

    def glue_all_params(self, glue_ref: str, transpose: bool = False, run: mlflow.entities.run.Run = None):
        """Glue all parameters as a dataframe table. Transpose option.
        By default for the configured latest run, but you are free to supply any Run object."""
        if run is None: run = self.latest_run
        params_df = self.get_params_as_df(run)
        if transpose: params_df = params_df.transpose()
        glue(glue_ref, params_df, display=False)

    def glue_all_metrics(self, glue_ref: str, transpose: bool = False, run: mlflow.entities.run.Run = None):
        """Glue all metrics as a dataframe table. Transpose option.
        By default for the configured latest run, but you are free to supply any Run object."""
        if run is None: run = self.latest_run
        metrics_df = self.get_metrics_as_df(run)
        if transpose: metrics_df = metrics_df.transpose()
        glue(glue_ref, metrics_df, display=False)

    def glue_image(self,
                   glue_ref: str,
                   image_path: Union[str, Path],
                   run: mlflow.entities.run.Run = None,
                   ):
        """Glues an image from a Run. By default the configured latest run, but you can supply any.
        Note that you can also get an image in your MyST notebooks by any of:
            1. ![](uri)
            2. ```{figure} uri```
            3. simply as output of a python cell
        Glueing provides some more options, but the above may be easier depending on your usecase.

        Args:
            run: any MLFlow Run object
            image_path: the relative path of the image within the Run artifacts. Usually something like 'images/myimage.png'
            glue_ref: name by which to refer to the glued object.

        Returns:
            -
        """
        if run is None: run = self.latest_run
        image_uri = Path(run.info.artifact_uri)/image_path
        if glue_ref is None: glue_ref=Path(image_path).stem
        glue_image_by_uri(glue_ref, image_uri)

    def glue_model_reference_metadata(self, glue_ref: str, run: mlflow.entities.run.Run = None):
        """Glue commit hash, run id and the run's end time as a dictionary.
        By default for the configured latest run, but you are free to supply any Run object.

        Args:
            run: any MLFlow Run object.
            glue_ref: name by which to refer to the glued object.

        Returns:
            -
        """
        if run is None: run = self.latest_run
        internal_meta_data = dict(
            commit_hash=run.data.tags.get('mlflow.source.git.commit', 'N/A'),
            run_id=run.info.run_id,
            run_end_time_local=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.info.end_time / 1000.)),
        )
        if glue_ref is None: glue_ref = 'model_ref_metadata'
        glue(glue_ref, internal_meta_data, display=False)

    def extract_journal_entry_data(self, run: mlflow.entities.run.Run = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """Gets some metadata, the parameters and metrics from the run and returns them in dict or dataframe objects.
        By default for the configured latest run, but you are free to supply any Run object.

        Args:
            run: any MLFlor Run object

        Returns: tuple of:
            dict of header,
            pandas dataframe of parameters,
            pandas dataframe of metrics
        """
        if run is None: run = self.latest_run
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
        """Wrapper for getting the parameters from a Run as dataframe

        Args:
            run: any MLFlow Run object
            drop: optional list of strings for parameters to exclude

        Returns:
            pandas dataframe with the parameters.
        """
        return get_params_as_df(run, drop=drop)

    def get_metrics_as_df(self,
                         run: mlflow.entities.run.Run,
                         drop: Optional[List[str]] = None
                         ) -> pd.DataFrame:
        """Wrapper for getting the metrics from a Run as dataframe

        Args:
            run: any MLFlow Run object
            drop: optional list of strings for metrics to exclude

        Returns:
            pandas dataframe with the metrics.
        """
        return get_metrics_as_df(run, drop=drop)


def glue_image_by_uri(glue_name, image_uri):
    """Glue image by absolute URI

    Args:
        glue_name: string
        image_uri: absolute path of the image file.

    Returns:
        -
    """
    im = Image.open(image_uri)
    glue(glue_name, im, display=False)


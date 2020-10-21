import pandas as pd
from matplotlib import pyplot as plt

from typing import List, Optional, NoReturn, Union

import mlflow
from mlflow.entities import ViewType
from mlflow.entities.run import Run

def set_note(note: str) -> NoReturn:
    mlflow.set_tag('mlflow.note.content', note)

def set_tags(tags: dict) -> NoReturn:
    for key, value in tags.items():
            mlflow.set_tag(key, value)

def log_fig(filename: str, fig=None) -> NoReturn:
    if not fig:
        fig = plt.gcf()
    fig.savefig(filename)
    mlflow.log_artifact(filename)




def get_experiment_id(experiment_name: str) -> str:
    return mlflow.get_experiment_by_name(experiment_name).experiment_id

def get_latest_run(experiment_id: Union[str, int],
                   tags: dict = None,
                   status: str = "FINISHED",
                   custom_query: str = None):
    """Get the latest MLFLow run that matched the parameters.

    Params:
        tags: dictionary of tagname, value pairs. Note that a run without a supplied tag will not get matched in any case.
        custom_query: string to be added to query in addition to tag and status clauses
    """
    query = f"attributes.status = '{status}'"
    if tags is not None:
        tags_query = [f"tags.`{key}` =  '{value}'" for key, value in tags.items()]
        tags_query = " and ".join(tags_query)
        query = f"{query} and {tags_query}"
    if custom_query is not None:
        query = f"{query} and {custom_query}"
    latest_run = mlflow.get_run(
        mlflow.search_runs(experiment_ids=[experiment_id],
                           run_view_type=ViewType.ACTIVE_ONLY,
                           filter_string=query,
                           max_results=1).loc[0].run_id
    )
    return latest_run


def get_params_as_df(run: Run,
                     drop: Optional[List[str]] = None
                     ) -> pd.DataFrame:
    params = run.data.params
    if drop:
        for key in drop: params.pop(key, None)
    return pd.DataFrame(params.items(), columns=['Parameter','Value'])

def get_metrics_as_df(run: Run,
                     drop: Optional[List[str]] = None
                     ) -> pd.DataFrame:
    metrics = run.data.metrics
    if drop:
        for key in drop: metrics.pop(key, None)
    return pd.DataFrame(metrics.items(), columns=['Metric','Value'])
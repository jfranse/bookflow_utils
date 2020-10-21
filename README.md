# BookFlow Utils

This is a helper package designed for use with the BookFlow way of working: [BookFlow Guide.](https://bookflow.jeroenfranse.com)

**Warning**, this package is currently in development

## Installation

Clone respository and run `python setup.py`

## Usage

### Helper Object with Gluing Shortcuts

Let's say you have a project for which you track the runs using MLFlow and these are stored at some location `/path/to/mlruns`. You are writing a report using the BookFlow method, and this report will only be about the results of the latest run.

However, during development you sometimes make some test runs, but the results from those are not to be taken seriously. You want to avoid mixing up testing runs with serious runs. Therefore, you will apply tags to your runs. For example, 'test_run' set to 'True' in the case of a test run, and 'test_run' set to 'False' for a serious run. 

In that case, all we have to do to make sure that our report always contains the results of the latest serious run is the following:

```python
from bookflow_utils.mlflow_to_book import BookflowConfig, BookflowHelper
config = BookflowConfig(tracking_uri='/path/to/mlruns',
                        experiment_name='experiment_name',
                        target_tag={'test_run': 'False'}) 
helper = BookflowHelper(config)

# at this point, our helper has already retreived the latest run, 
# so we can start gluing whatever we want

helper.glue_param('algorithm', 'algo') # a single parameter
helper.glue_metric('accuracy', 'acc_test') # a single metrics
helper.glue_all_params('params') # a table of all parameters
helper.glue_all_metrics('metrics') # a table of all metrics
helper.glue_image('learning_curve', 'plots/learning_curve.png') # a figure
helper.glue_model_reference_metadata('metadata') # a table of run metadata
```

Note that any runs _without_ the tag 'test_run', will _not_ be matched by the helper.

All gluing functions take as the first argument the Jupyter Book reference (you'll use this name inside your report), and sometimes as the second argument the name of the metric or parameter or image inside mlflow.

#### Gluing Multiple Runs

Perhaps your report is a little fancier, and you want to refer to multiple runs, perhaps to compare them. This is easily done:

- distinguish your types of runs inside MLFLow by separating into different experiments, or by supplying different tags
- create another instance of the BookflowHelper objects, specifying the different experiment and/or tag(s)

If your situation is more complicated, you may be helped by either of the following
- the `get_latest_run()` method of the helper object can be used to supply alternative tags and an additional custom query string (works just like the MLFlow search bar) and returns the latest run object matching that query, which can then be used as described in the next point
- all gluing functions take an optional `run=` argument that takes any MLFlow Run object that you care to provide, and will ignore the rest of the helper's configuration for that call only

### CLI

Some occasionally useful functionality is included in a straightforward CLI. 
- to create a new MyST `.md` file that can immediately be opened in jupyter notebook and is already synced by jupytext `bookflow create myst path/to/newfile.md`
- to dump a yaml config file template for use with the BookflowConfig object `bookflow create config path/to/config.yaml`
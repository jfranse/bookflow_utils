import click
from pathlib import Path
from shutil import copyfile
import yaml
import jinja2, jupytext, myst_parser
from bookflow_utils.mlflow_to_book import BookflowConfig

@click.group()
def cli():
    pass

@click.group()
def create():
    pass

@create.command()
@click.argument('filepath')
def myst(filepath):
    """Create a new MyST markdown file with the appropriate preamble for an ipython3 notebook."""
    target_path = Path(filepath).parent
    target_name = Path(filepath).stem
    if '.md' not in target_name: target_name = target_name+'.md'
    click.echo(f"Creating new MyST markdown file {target_name} in location: {target_path}")

    preamble_location = Path(__file__).parent/Path('resources/')
    preamble_name = 'myst_ipython3_preamble.j2'

    preamble_data = dict(
        jupytext_version = jupytext.__version__,
        myst_parser_version = '.'.join(myst_parser.__version__.split('.')[0:2]),
    )

    templateLoader = jinja2.FileSystemLoader(searchpath=preamble_location)
    templateEnv = jinja2.Environment(loader=templateLoader, autoescape=True)
    template = templateEnv.get_template(preamble_name)
    output_preamble = template.render(preamble_data)

    with open(target_path/target_name, 'w') as f:
        f.write(output_preamble)

@create.command()
def labjournal():
    """NOT IMPLEMENTED YET"""
    pass

@create.command()
@click.argument('filepath')
def config(filepath):
    """Dump a template of a bookflow config file"""
    target = Path(filepath)
    if target.exists():
        click.echo(f"File {target} already exists, NOT overwriting")
    else:
        with open(target, 'w') as stream:
            yaml.safe_dump(BookflowConfig.config_template, stream)


cli.add_command(create)
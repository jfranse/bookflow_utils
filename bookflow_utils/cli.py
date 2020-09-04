import click
from pathlib import Path
from shutil import copyfile
import yaml
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
    preamble_file = Path(__file__).parent/Path('resources/myst_ipython3_preamble.txt')
    copyfile(preamble_file, target_path/target_name)

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
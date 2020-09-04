from setuptools import setup

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setup(
   name='bookflow_utils',
   version='0.1',
   description='Utilities to help with BookFlow',
   author='Jeroen Franse',
   author_email='contact@jeroenfranse.com',
   packages=['bookflow_utils'],  #same as name
   install_requires=['mlflow', 'jupyter-book', 'click'], #external packages as dependencies
   include_package_data=True,
   entry_points='''
        [console_scripts]
        bookflow=bookflow_utils.cli:cli
   '''
)
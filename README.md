
# External library examples for ``RESSPECT``


[![resspect](https://img.shields.io/badge/COIN--Focus-RESSPECT-red)](http://cosmostatistics-initiative.org/resspect/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lsstdesc/resspect_external/smoke-test.yml)](https://github.com/lsstdesc/resspect_external/actions/workflows/smoke-test.yml)
[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

<!-- [![Codecov](https://codecov.io/gh/lsstdesc/resspect_external/branch/main/graph/badge.svg)](https://codecov.io/gh/lsstdesc/resspect_external)
[![Read The Docs](https://img.shields.io/readthedocs/resspect-external)](https://resspect-external.readthedocs.io/) -->


This collection of example classes demonstrates the minimum code needed to define
the various plugin classes that ``RESSPECT`` supports. For more information about
``RESSPECT`` please see the following:
* Repository: https://github.com/lsstdesc/resspect
* ReadTheDocs: https://lsst-resspect.readthedocs.io/en/latest/

Feel free to copy or fork this repository as a way to get started using an externally
defined feature extractor, classifier, or query strategy for ``RESSPECT``.

Note that it is not a requirement to implement all three example classes, they
are simply grouped together here to be concise. Implementing only one or two of
the different classes shown here is completely acceptable.

## Example usage of externally library

``RESSPECT`` has two primary function to drive active learning:
* ``learn_loop`` ([link](https://github.com/LSSTDESC/RESSPECT/blob/6e1396bdf83c495fea3f9752887db8fcd36b68b0/src/resspect/learn_loop.py#L288))
* ``time_domain_loop`` ([link](https://github.com/LSSTDESC/RESSPECT/blob/6e1396bdf83c495fea3f9752887db8fcd36b68b0/src/resspect/time_domain_loop.py#L805))

To define the use of an external class and organize the large number of input
parameters necessary for each of these a configuration class is used, either:
``LoopConfiguration`` ([link](https://github.com/LSSTDESC/RESSPECT/blob/6e1396bdf83c495fea3f9752887db8fcd36b68b0/src/resspect/loop_configuration.py#L8)) for ``learn_loop`` or,
``TimeDomainConfiguration`` ([link](https://github.com/LSSTDESC/RESSPECT/blob/6e1396bdf83c495fea3f9752887db8fcd36b68b0/src/resspect/time_domain_configuration.py#L8C7-L8C30)) for ``time_domain_loop``.

To make use of an externally defined feature extractor, classifier, or query
strategy, simply define the class to be used in the configuration object. The
following example shows how to use the libpath to specify which of the external
classes ``RESSPECT`` should use when running ``learn_loop``.

```python
from resspect.loop_configuration import LoopConfiguration
from resspect.learn_loop import learn_loop

# Define the loop configuration
loop_config = LoopConfiguration(
    # ... preceding parameters
    strategy='resspect_external.example_query_strategy.ExampleQueryStrategy',
    classifier='resspect_external.example_classifier.ExampleClassifier',
    features_method='resspect_external.example_feature_extractor.ExampleFeatureExtractor',
    # ... additional parameters
)

# Use loop configuration to call active learning loop
learn_loop(loop_config)
```

Note in this example code, we instruct ``RESSPECT`` to make use of the
``ExampleFeatureExtractor`` as well as the ``ExampleClassifier`` and ``ExampleQueryStrategy``
classes by providing the libpath string as the parameters.

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create -n <env_name> python=3.11
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> ./.setup_dev.sh
>> conda install pandoc
```

Notes:
1. `./.setup_dev.sh` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
2. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)


## Acknoledgements

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

# TraP_ML_tools
Python tools and scripts to reproduce results in Rowlinson et al. (in prep)

This repository contains all the scripts required to reproduce the analysis in Rowlinson et al. (in prep). Many of the machine learning tools are not TraP specific, but the examples utilise the TraP databases.

Acknowledgements
----------------

This work utilises a number of Python libraries, including the Matplotlib plotting libraries Hunter (2007, Computing in Science & Engineering, 9, 90-95) and astroML (Vanderplas et al., 2012, in Conference on Intelligent Data Understanding, 47-–54). Finally, the authors acknowledge training in machine learning strategies received from the Stanford Machine Learning course available on Coursera (https://www.coursera.org/course/ml).

Requirements
------------

- tkp (The LOFAR Transients Pipeline, TraP, https://github.com/transientskp/tkp)
- sqlalchemy
- scipy
- numpy
- pandas
- logging
- sys
- matplotlib
- pylab
- os
- glob
- random
- math
- astroML
- multiprocessing
- operator
- Jupyter Notebook

Contents
--------

#### database_tools
[dbtools.py](https://github.com/AntoniaR/TraP_ML_tools/database_tools/dbtools.py)
The tools required to access the TraP databases, you will need a dblogin.py from the examples folder and to fill in the appropriate fields for your database.

#### examples
[FilterVariables.py](https://github.com/AntoniaR/TraP_ML_tools/examples/FilterVariables.py)  & [FilterVariables.ipynb](https://github.com/AntoniaR/TraP_ML_tools/examples/FilterVariables.ipynb)
Scripts to produce figures 3 and 4 in Rowlinson et al. (in prep) and can conduct a simple sigma threshold. Requires data processed by the TraP database. Available as both a python script and a Jupyter Notebook.

[anomaly_detection.py](<https://github.com/AntoniaR/TraP_ML_tools/examples/anomaly_detection.py) & [anomaly_detection.ipynb](https://github.com/AntoniaR/TraP_ML_tools/examples/anomaly_detection.ipynb)
Scripts to train the anomaly detection algorithm  (Section 4.1.1 in Rowlinson et al. in prep). Requires data from [ml_csv_files](https://github.com/AntoniaR/TraP_ML_tools/examples/ml_csv_files). Available as both a python script and a Jupyter Notebook.

[dblogin.py](https://github.com/AntoniaR/TraP_ML_tools/examples/dblogin.py)
Input your database information here to be able to use the dbtools.py script from database_tools.

[logistic_regreassion.py](https://github.com/AntoniaR/TraP_ML_tools/examples/logistic_regression.py) & [logistic_regreassion.ipynb](https://github.com/AntoniaR/TraP_ML_tools/examples/logistic_regression.ipynb)
Scripts to train the logistic regression algorithm  (Section 4.1.2 in Rowlinson et al. in prep). Requires data from folder [ml_csv_files](https://github.com/AntoniaR/TraP_ML_tools/examples/ml_csv_files). Available as both a python script and a Jupyter Notebook.

[machine_learning_tests.py](https://github.com/AntoniaR/TraP_ML_tools/examples/machine_learning_tests.py)
Scripts to test the machine learning algorithms  (Section 4.2 in Rowlinson et al. in prep). Requires data from folder [ml_csv_files](https://github.com/AntoniaR/TraP_ML_tools/examples/ml_csv_files). Not available as an ipynb due to the long run time for this script.

[transient_margins.py](https://github.com/AntoniaR/TraP_ML_tools/examples/transient_margins.py) & [transient_margins.ipynb](https://github.com/AntoniaR/TraP_ML_tools/examples/transient_margins.ipynb)
Scripts to train the sigma margin algorithm  (Section 4.1.3 in Rowlinson et al. in prep). Requires data from folder [ml_csv_files](https://github.com/AntoniaR/TraP_ML_tools/examples/ml_csv_files). Available as both a python script and a Jupyter Notebook.

#### machine_learning
[MLtests.py](https://github.com/AntoniaR/TraP_ML_tools/machine_learning/MLtests.py)
Tools to conduct the machine learning tests outlined in Rowlinson et al. (in prep) section 4.2

[generic_tools.py](https://github.com/AntoniaR/TraP_ML_tools/machine_learning/generic_tools.py)
Various tools used by the machine learning algorithms

[plotting_tools.py](https://github.com/AntoniaR/TraP_ML_tools/machine_learning/plotting_tools.py)
Some plotting tools used to create the figures in Rowlinson et al. (in prep)

[train_anomaly_detect](https://github.com/AntoniaR/TraP_ML_tools/machine_learning/train_anomaly_detect.py)
Tools to train and use an anomaly detection algorithm. (Section 4.1.1 in Rowlinson et al. in prep)

[train_logistic_regression.py](https://github.com/AntoniaR/TraP_ML_tools/machine_learning/train_logistic_regression.py)
Tools to train and use a logistic regression algorithm (Section 4.1.2 in Rowlinson et al. in prep)

[train_sigma_margin.py](https://github.com/AntoniaR/TraP_ML_tools/machine_learning/train_sigma_margin.py)
Tools to train and use the sigma margin algorithm (Section 4.1.3 in Rowlinson et al. in prep)

#### plotting
[plot_varib_params.py](https://github.com/AntoniaR/TraP_ML_tools/plotting/plot_varib_params.py)
Tools required to create the plots throughout Rowlinson et al. (in prep)

#### tools
[tools.py](https://github.com/AntoniaR/TraP_ML_tools/machine_learning/tools.py)
A few generic, useful tools.

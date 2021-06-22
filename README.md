This repository consists of the implementations of the proposed EBCR method, which aims to improve similarity measurement used in memory-based collaborative filtering recommendation systems.

To run the method, the [Surprise library](http://surpriselib.com/) needs to be installed:
  -	pip install numpy
  -	pip install scikit-surprise

After that the Surprise library is installed, run the method by:
  -	python main.py

This command will execute the proposed method (and its variants) on several benchmarks available in the [Surprise library](http://surpriselib.com/). Note that by running this script at the first time, you will be requested to confirm the download of the corresponding rating datasets, which is the fonctionality provided by the Surprise library. For people who would like to manually download the datasets, we also provided the corresonding files in the "datasets" folder.

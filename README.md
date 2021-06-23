This repository consists of the implementations of the proposed EBCR method, which aims to improve similarity measurement used in memory-based collaborative filtering recommendation systems.

To run the method, the [Surprise library](http://surpriselib.com/) needs to be installed:
  -	pip install numpy
  -	pip install scikit-surprise

After that the Surprise library is installed, run the method by:
  -	python main.py

This command will execute the proposed method (and its variants) on several benchmarks available in the [Surprise library](http://surpriselib.com/). Note that by running this script at the first time, you will be requested to confirm the download of the corresponding rating datasets, which is the fonctionality provided by the Surprise library. For people who would like to manually download the datasets, we also provided the corresonding files in the "datasets" folder.

### Overall of EBCR method

The EBCR term is the abbreviation of "Empirical Bayes Concordance Ratio", which is composed of two parts: the CR part and the EB part. The aim of the CR part is to eliminate the disparity of users' rating behaviors. This is done by first relaxing (discretizing) user tastes into three classes (i.e. like, neutral and dislike) and then by computing the ratio of users' concordantly co-rated items to model the rating concordance between users. The EB part is then used to adjust these ratios by considering the ratio distribution within the whole training samples. The adjusted ratios are then used to weight similarity measurement between users and/or items during a typical memory-based CF approach.

### Results overview
The results of the experimentations are based on three benchmark datasets: [MovieLens-100K](https://grouplens.org/datasets/movielens/), [MovieLens-1M](https://grouplens.org/datasets/movielens/) and [Jester](https://grouplens.org/datasets/jester/).

#### Comparing with state-of-the-art CF models.

|          | MovieLens 100k | MovieLens 100k | MovieLens 1M | MovieLens 1M | Jester | Jester |
|:--------:|:--------------:|:--------------:|:------------:|:------------:|:------:|:------:|
| Approach |       MAE      |      RMSE      |      MAE     |     RMSE     |   MAE  |  RMSE  |
| Baseline |     0.7484     |      0.944     |    0.7195    |    0.9088    | 3.3982 | 4.3134 |
|    SVD   |     0.7376     |     <ins>0.9358</ins>     |    0.6863    |    <ins>0.8743</ins>    | 3.3713 | 4.5004 |
|   SVD++  |     **0.7214**     |     **0.9203**     |    **0.6729**    |    **0.8625**    | 3.6209 | 4.9042 |
|   NeuMF  |     0.7437     |     0.9363     |    <ins>0.6773</ins>    |    0.8765    | <ins>3.0375</ins> | <ins>4.1376</ins> |
|   EBCR   |     <ins>0.7348</ins>     |     0.9413     |    0.7052    |    0.9016    | **3.0158** | **4.1008** |

The "results" folder contains the details of evaluation results comparing with other methods.

The datasets are also available in the public [zenodo repository](https://doi.org/10.5281/zenodo.5013115).

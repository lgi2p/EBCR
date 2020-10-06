import numpy as np
import csv

from surprise import Dataset, KNNBasic, SVD, SVDpp, BaselineOnly
from surprise.model_selection import KFold, cross_validate
from cf_models import EbcrMsdKNN, EbcrCosKNN, EbcrNormPccKNN, NormPcc, SW_Norm_PccKNN, SW_MSD_KNN, SW_COS_KNN, LS_MSD_KNN, LS_COS_KNN, LS_Norm_PccKNN

__author__ = "Yu DU"

# Datasets initialisation
ml_100k = Dataset.load_builtin('ml-100k')
ml_1m = Dataset.load_builtin('ml-1m')
jester = Dataset.load_builtin('jester')

# Split train and test set
kf = KFold(random_state=0, n_splits=5)

list_k = [5, 10, 20, 40, 60, 80, 100, 200]
list_k2 = [5, 10, 15, 20, 25, 30, 35, 40]

# The Ml-100k Dataset
with open('results/ml100k_all.csv', mode='w') as result_file:
    fieldnames = ['k', 'algo', 'MAE', 'RMSE']
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()

    # SVD algo
    svd = SVD()
    out_svd = cross_validate(svd, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'SVD',
                'MAE': round(float(np.mean(out_svd['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_svd['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    # SVD++ algo
    svdpp = SVDpp()
    out_svdpp = cross_validate(svdpp, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'SVD++',
                'MAE': round(float(np.mean(out_svdpp['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_svdpp['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    # Baseline algo
    baseline = BaselineOnly()
    out_bl = cross_validate(baseline, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'Baseline',
                'MAE': round(float(np.mean(out_bl['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_bl['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    for k in list_k:
        # KNN with MSD as sim_metric
        msd_knn = KNNBasic(k=k, sim_options={'name': 'msd', 'user_based': True})
        out_msd_knn = cross_validate(msd_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'MSD_KNN',
                    'MAE': round(float(np.mean(out_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_msd_knn = SW_MSD_KNN(k=k, sim_options={'user_based': True})
        out_sw_msd_knn = cross_validate(sw_msd_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_MSD_KNN',
                    'MAE': round(float(np.mean(out_sw_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_msd_knn = LS_MSD_KNN(k=k, sim_options={'user_based': True})
        out_bcr_msd_knn = cross_validate(bcr_msd_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_MSD_KNN',
                    'MAE': round(float(np.mean(out_bcr_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_msd_knn = EbcrMsdKNN(k=k, sim_options={'name': 'msd', 'user_based': True}, min_freq=20)
        out_ebcr_msd_knn = cross_validate(ebcr_msd_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_MSD_KNN',
                    'MAE': round(float(np.mean(out_ebcr_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        # KNN with cosine as sim_metric
        cos_knn = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': True})
        out_cos_knn = cross_validate(cos_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'COS_KNN',
                    'MAE': round(float(np.mean(out_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_cos_knn = SW_COS_KNN(k=k, sim_options={'user_based': True})
        out_sw_cos_knn = cross_validate(sw_cos_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_COS_KNN',
                    'MAE': round(float(np.mean(out_sw_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_cos_knn = LS_COS_KNN(k=k, sim_options={'user_based': True})
        out_bcr_cos_knn = cross_validate(bcr_cos_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_COS_KNN',
                    'MAE': round(float(np.mean(out_bcr_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_cos_knn = EbcrCosKNN(k=k, sim_options={'user_based': True}, min_freq=20)
        out_ebcr_cos_knn = cross_validate(ebcr_cos_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_COS_KNN',
                    'MAE': round(float(np.mean(out_ebcr_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        # KNN with pcc as sim_metric
        normpcc_knn = NormPcc(k=k, sim_options={'user_based': True})
        out_normpcc_knn = cross_validate(normpcc_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'NormPCC_KNN',
                    'MAE': round(float(np.mean(out_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_normpcc_knn = SW_Norm_PccKNN(k=k, sim_options={'user_based': True})
        out_sw_normpcc_knn = cross_validate(sw_normpcc_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_sw_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_normpcc_knn = LS_Norm_PccKNN(k=k, sim_options={'user_based': True})
        out_bcr_normpcc_knn = cross_validate(bcr_normpcc_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_bcr_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_normpcc_knn = EbcrNormPccKNN(k=k, sim_options={'user_based': True}, min_freq=20)
        out_ebcr_normpcc_knn = cross_validate(ebcr_normpcc_knn, ml_100k, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_ebcr_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

# The ml1m Dataset
with open('results/ml1m_all.csv', mode='w') as result_file:
    fieldnames = ['k', 'algo', 'MAE', 'RMSE']
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()

    # SVD algo
    svd = SVD()
    out_svd = cross_validate(svd, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'SVD',
                'MAE': round(float(np.mean(out_svd['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_svd['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    # SVD++ algo
    svdpp = SVDpp()
    out_svdpp = cross_validate(svdpp, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'SVD++',
                'MAE': round(float(np.mean(out_svdpp['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_svdpp['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    # Baseline algo
    baseline = BaselineOnly()
    out_bl = cross_validate(baseline, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'Baseline',
                'MAE': round(float(np.mean(out_bl['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_bl['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    for k in list_k:
        # KNN with MSD as sim_metric
        msd_knn = KNNBasic(k=k, sim_options={'name': 'msd', 'user_based': True})
        out_msd_knn = cross_validate(msd_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'MSD_KNN',
                    'MAE': round(float(np.mean(out_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_msd_knn = SW_MSD_KNN(k=k, sim_options={'user_based': True})
        out_sw_msd_knn = cross_validate(sw_msd_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_MSD_KNN',
                    'MAE': round(float(np.mean(out_sw_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_msd_knn = LS_MSD_KNN(k=k, sim_options={'user_based': True})
        out_bcr_msd_knn = cross_validate(bcr_msd_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_MSD_KNN',
                    'MAE': round(float(np.mean(out_bcr_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_msd_knn = EbcrMsdKNN(k=k, sim_options={'name': 'msd', 'user_based': True}, min_freq=60)
        out_ebcr_msd_knn = cross_validate(ebcr_msd_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_MSD_KNN',
                    'MAE': round(float(np.mean(out_ebcr_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        # KNN with cosine as sim_metric
        cos_knn = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': True})
        out_cos_knn = cross_validate(cos_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'COS_KNN',
                    'MAE': round(float(np.mean(out_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_cos_knn = SW_COS_KNN(k=k, sim_options={'user_based': True})
        out_sw_cos_knn = cross_validate(sw_cos_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_COS_KNN',
                    'MAE': round(float(np.mean(out_sw_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_cos_knn = LS_Norm_PccKNN(k=k, sim_options={'user_based': True})
        out_bcr_cos_knn = cross_validate(bcr_cos_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_COS_KNN',
                    'MAE': round(float(np.mean(out_bcr_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_cos_knn = EbcrCosKNN(k=k, sim_options={'user_based': True}, min_freq=60)
        out_ebcr_cos_knn = cross_validate(ebcr_cos_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_COS_KNN',
                    'MAE': round(float(np.mean(out_ebcr_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        # KNN with pcc as sim_metric
        normpcc_knn = NormPcc(k=k, sim_options={'user_based': True})
        out_normpcc_knn = cross_validate(normpcc_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'NormPCC_KNN',
                    'MAE': round(float(np.mean(out_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_normpcc_knn = SW_Norm_PccKNN(k=k, sim_options={'user_based': True})
        out_sw_normpcc_knn = cross_validate(sw_normpcc_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_sw_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_normpcc_knn = LS_Norm_PccKNN(k=k, sim_options={'user_based': True})
        out_bcr_normpcc_knn = cross_validate(bcr_normpcc_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_bcr_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_normpcc_knn = EbcrNormPccKNN(k=k, sim_options={'user_based': True}, min_freq=60)
        out_ebcr_normpcc_knn = cross_validate(ebcr_normpcc_knn, ml_1m, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_ebcr_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

# The jester Dataset
with open('results/jester_all.csv', mode='w') as result_file:
    fieldnames = ['k', 'algo', 'MAE', 'RMSE']
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()

    # SVD algo
    svd = SVD()
    out_svd = cross_validate(svd, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'SVD',
                'MAE': round(float(np.mean(out_svd['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_svd['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    # SVD++ algo
    svdpp = SVDpp()
    out_svdpp = cross_validate(svdpp, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'SVD++',
                'MAE': round(float(np.mean(out_svdpp['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_svdpp['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    # Baseline algo
    baseline = BaselineOnly()
    out_bl = cross_validate(baseline, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
    raw_dict = {'k': 'no_k',
                'algo': 'Baseline',
                'MAE': round(float(np.mean(out_bl['test_mae'])), 4),
                'RMSE': round(float(np.mean(out_bl['test_rmse'])), 4)}
    writer.writerow(raw_dict)

    for k in list_k2:
        # KNN with MSD as sim_metric
        msd_knn = KNNBasic(k=k, sim_options={'name': 'msd', 'user_based': False})
        out_msd_knn = cross_validate(msd_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'MSD_KNN',
                    'MAE': round(float(np.mean(out_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_msd_knn = SW_MSD_KNN(k=k, sim_options={'user_based': False})
        out_sw_msd_knn = cross_validate(sw_msd_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_MSD_KNN',
                    'MAE': round(float(np.mean(out_sw_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_msd_knn = LS_MSD_KNN(k=k, sim_options={'user_based': False})
        out_bcr_msd_knn = cross_validate(bcr_msd_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_MSD_KNN',
                    'MAE': round(float(np.mean(out_bcr_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_msd_knn = EbcrMsdKNN(k=k, sim_options={'name': 'msd', 'user_based': False}, min_freq=150)
        out_ebcr_msd_knn = cross_validate(ebcr_msd_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_MSD_KNN',
                    'MAE': round(float(np.mean(out_ebcr_msd_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_msd_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        # KNN with cosine as sim_metric
        cos_knn = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': False})
        out_cos_knn = cross_validate(cos_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'COS_KNN',
                    'MAE': round(float(np.mean(out_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_cos_knn = SW_COS_KNN(k=k, sim_options={'user_based': False})
        out_sw_cos_knn = cross_validate(sw_cos_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_COS_KNN',
                    'MAE': round(float(np.mean(out_sw_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_cos_knn = LS_COS_KNN(k=k, sim_options={'user_based': False})
        out_bcr_cos_knn = cross_validate(bcr_cos_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_COS_KNN',
                    'MAE': round(float(np.mean(out_bcr_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_cos_knn = EbcrCosKNN(k=k, sim_options={'user_based': False}, min_freq=150)
        out_ebcr_cos_knn = cross_validate(ebcr_cos_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_COS_KNN',
                    'MAE': round(float(np.mean(out_ebcr_cos_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_cos_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        # KNN with pcc as sim_metric
        normpcc_knn = NormPcc(k=k, sim_options={'user_based': False})
        out_normpcc_knn = cross_validate(normpcc_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'NormPCC_KNN',
                    'MAE': round(float(np.mean(out_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        sw_normpcc_knn = SW_Norm_PccKNN(k=k, sim_options={'user_based': False})
        out_sw_normpcc_knn = cross_validate(sw_normpcc_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'SW_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_sw_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_sw_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        bcr_normpcc_knn = LS_Norm_PccKNN(k=k, sim_options={'user_based': False})
        out_bcr_normpcc_knn = cross_validate(bcr_normpcc_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'BCR_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_bcr_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_bcr_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

        ebcr_normpcc_knn = EbcrNormPccKNN(k=k, sim_options={'user_based': False}, min_freq=150)
        out_ebcr_normpcc_knn = cross_validate(ebcr_normpcc_knn, jester, ['rmse', 'mae'], kf, verbose=False, n_jobs=-1)
        raw_dict = {'k': k,
                    'algo': 'EBCR_NormPCC_KNN',
                    'MAE': round(float(np.mean(out_ebcr_normpcc_knn['test_mae'])), 4),
                    'RMSE': round(float(np.mean(out_ebcr_normpcc_knn['test_rmse'])), 4)}
        writer.writerow(raw_dict)

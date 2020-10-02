import numpy as np
from six import iteritems
import heapq


from surprise import KNNBasic
from surprise import AlgoBase, PredictionImpossible
from scipy.stats import beta

__author__ = "Yu DU"


class EbcrMsdKNN(KNNBasic):
    """
    The variant of a basic collaborative filtering algorithm using mean squared distance measure adjusted by EBCR ratio.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation. Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
        alpha0(double), beta0(double): The hyper-parameters of the distribution of concordance ratios, they will be estimated
            by maximum likelihood estimator during the training phase.
        normalise_rating(bool): Whether to use normalized ratings during the prediction phase.
    """

    def __init__(self, k=40, min_k=1, a=0.5, sim_options={}, alpha0=0.0, beta0=0.0, min_freq=0, normalise_rating=True, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.all_user_info = {}
        self.normalized_rating = {}
        self.normalise_rating = normalise_rating
        self.min_freq = min_freq
        self.a = a

    def fit(self, trainset):
        """
        The fit function calculate the similarity matrix for the given train set
        """
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        self.bu, self.bi = self.compute_baselines()
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur
        # print(n_x)
        freq = np.zeros((n_x, n_x), np.int)
        concordant_freq = np.zeros((n_x, n_x), np.int)
        sq_diff = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        self.all_user_info, self.normalized_item_rating = self.normalize_user_rating(trainset)

        for item, y_ratings in iteritems(self.normalized_item_rating):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    sq_diff[xi, xj] += (ri - rj) ** 2
                    freq[xi, xj] += 1
                    if self.is_concordant(ri, rj):
                        concordant_freq[xi, xj] += 1

        concordant_ratio_list = list()
        for xi in range(n_x):
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] > self.min_freq: #tune nb co-rated item for large dataset
                    ratio = concordant_freq[xi, xj] / freq[xi, xj]
                    if ratio == 0:
                        ratio = 0.001
                    if ratio == 1:
                        ratio = 0.999
                    concordant_ratio_list.append(ratio)
        # print(len(concordant_ratio_list))

        # if self.alpha0 == 0.0 and self.beta0 == 0.0:
        alpha0, beta0, _, __ = beta.fit(concordant_ratio_list, floc=0, fscale=1)
        self.alpha0 = alpha0
        self.beta0 = beta0
        # print(alpha0, beta0)

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                # norm_sq_diff = sq_diff[xi, xj] / freq[xi, xj]
                if freq[xi, xj] == 0:
                    sim[xi, xj] = 0
                else:
                    inverse_diff = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)
                    sim[xi, xj] = ((concordant_freq[xi, xj] + self.alpha0) / (freq[xi, xj] + self.alpha0 + self.beta0)) \
                                  * inverse_diff
                sim[xj, xi] = sim[xi, xj]
        self.sim = sim
        # print(sim)
        return self

    def like_item_by_std(self, normalized_r_u, std):
        """
        determine the user taste given the centered and reduced rating and the standard deviation
        """
        if(normalized_r_u > std):
            return 1
        if (normalized_r_u < -std):
            return -1
        return 0

    def is_concordant(self, normalized_r_u1, normalized_r_u2):
        """
        function to determine whether or not two ratings are concordant
        """
        return self.like_item_by_std(normalized_r_u1, self.a) == self.like_item_by_std(normalized_r_u2, self.a)

    def normalize_user_rating(self, trainset):
        """
        function to normalize user ratings according to their own rating systems
        """
        ub = self.sim_options['user_based']

        overall_sigma = np.std([r for (_, _, r)
                                in trainset.all_ratings()])
        user_rating = trainset.ur if ub else trainset.ir
        # item_rating = dataset.yr
        all_user_info = {}
        normalized_item_rating = {}
        item_ratings = trainset.ir if ub else trainset.ur
        for item in item_ratings.keys():
            #for every item as key in a dict, construct a empty list
            normalized_item_rating[item] = list()

        for user, item_rating in iteritems(user_rating):
            user_info = {}
            user_mean = np.mean([rating for (item, rating) in item_rating])
            sigma = np.std([rating for (item, rating) in item_rating])
            user_std = overall_sigma if sigma == 0 else sigma
            user_normalise_r_value = [(item, (rating - user_mean) / user_std) for (item, rating) in item_rating]
            user_normalise_r_value_dict = {}

            for item, rating in user_normalise_r_value:
                user_normalise_r_value_dict[item] = rating

            for (item, norm_rating) in user_normalise_r_value:
                normalized_item_rating[item].append((user, norm_rating))
            user_info['mean_r'] = user_mean
            user_info['std_r'] = user_std
            user_info['normalized_r'] = user_normalise_r_value_dict
            all_user_info[user] = user_info

        return (all_user_info, normalized_item_rating)

    def estimate(self, u, i):
        """
        function to predict the rating value of the given user and item
        """

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        sum_sim = sum_ratings = actual_k = 0

        for (sim, r, neighbor_inner_id) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                r_normalise_neighbor = self.all_user_info[neighbor_inner_id]['normalized_r'][y]
                if self.normalise_rating:
                    sum_ratings += sim * r_normalise_neighbor
                else:
                    sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            est = self.trainset.global_mean + self.bu[x] + self.bi[y]
        else:
            if self.normalise_rating:
                est = self.all_user_info[x]['mean_r'] + (sum_ratings / sum_sim) * self.all_user_info[x]['std_r']
            else:
                est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class EbcrCosKNN(KNNBasic):

    """
    The variant of a basic collaborative filtering algorithm using cosine metric adjusted by EBCR ratio.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation. Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
        alpha0(double), beta0(double): The hyper-parameters of the distribution of concordance ratios, they will be estimated
            by maximum likelihood estimator during the training phase.
        normalise_rating(bool): Whether to use normalized ratings during the prediction phase.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, alpha0=0.0, beta0=0.0, min_freq=0, concordant_only=True, normalise_rating=True, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.all_user_info = {}
        self.normalized_rating = {}
        self.concordant_only = concordant_only
        self.normalise_rating = normalise_rating
        self.min_freq = min_freq

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        self.bu, self.bi = self.compute_baselines()
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        prods = np.zeros((n_x, n_x), np.double)
        freq = np.zeros((n_x, n_x), np.int)
        concordant_freq = np.zeros((n_x, n_x), np.int)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        self.all_user_info, self.normalized_item_rating = self.normalize_user_rating(trainset)

        for item, y_ratings in iteritems(self.normalized_item_rating):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    prods[xi, xj] += ri * rj
                    sqi[xi, xj] += ri ** 2
                    sqj[xi, xj] += rj ** 2
                    freq[xi, xj] += 1
                    if self.is_concordant(ri, rj):
                        concordant_freq[xi, xj] += 1

        concordant_ratio_list = list()
        for xi in range(n_x):
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] > self.min_freq: #tune nb co-rated item for large dataset
                    ratio = concordant_freq[xi, xj] / freq[xi, xj]
                    if ratio == 0:
                        ratio = 0.001
                    if ratio == 1:
                        ratio = 0.999
                    concordant_ratio_list.append(ratio)
        # print(len(concordant_ratio_list))

        # if self.alpha0 == 0.0 and self.beta0 == 0.0:
        alpha0, beta0, _, __ = beta.fit(concordant_ratio_list, floc=0, fscale=1)
        self.alpha0 = alpha0
        self.beta0 = beta0

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                # norm_sq_diff = sq_diff[xi, xj] / freq[xi, xj]
                if freq[xi, xj] == 0:
                    sim[xi, xj] = 0
                else:
                    denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                    sim[xi, xj] = ((concordant_freq[xi, xj] + self.alpha0) / (freq[xi, xj] + self.alpha0 + self.beta0)) \
                                  * (prods[xi, xj] / denum)

                sim[xj, xi] = sim[xi, xj]
        self.sim = sim

        return self

    def like_item_by_std (self, normalized_r_u, std):
        if(normalized_r_u > std): ##  evaluate parameter (std)
            return 1
        if (normalized_r_u < -std):
            return -1
        return 0

    def is_concordant(self, normalized_r_u1, normalized_r_u2):
        return self.like_item_by_std(normalized_r_u1, 0.5) == self.like_item_by_std(normalized_r_u2, 0.5)

    def normalize_user_rating(self, trainset):
        ub = self.sim_options['user_based']
        overall_sigma = np.std([r for (_, _, r)
                                in trainset.all_ratings()])
        user_rating = trainset.ur if ub else trainset.ir
        # item_rating = dataset.yr
        all_user_info = {}
        normalized_item_rating = {}
        item_ratings = trainset.ir if ub else trainset.ur
        for item in item_ratings.keys():
            #for every item as key in a dict, construct a empty list
            normalized_item_rating[item] = list()

        for user, item_rating in iteritems(user_rating):
            user_info = {}
            user_mean = np.mean([rating for (item, rating) in item_rating])
            sigma = np.std([rating for (item, rating) in item_rating])
            user_std = overall_sigma if sigma == 0 else sigma
            user_normalise_r_value = [(item, (rating - user_mean) / user_std) for (item, rating) in item_rating]
            user_normalise_r_value_dict = {}

            for item, rating in user_normalise_r_value:
                user_normalise_r_value_dict[item] = rating

            for (item, norm_rating) in user_normalise_r_value:
                normalized_item_rating[item].append((user, norm_rating))
            user_info['mean_r'] = user_mean
            user_info['std_r'] = user_std
            user_info['normalized_r'] = user_normalise_r_value_dict
            all_user_info[user] = user_info

        return (all_user_info, normalized_item_rating)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        # k_neighbors.pop(0) #delete the neighbor who is the user itself

        # print(k_neighbors)
        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0

        for (sim, r, neighbor_inner_id) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                r_normalise_neighbor = self.all_user_info[neighbor_inner_id]['normalized_r'][y]
                if self.normalise_rating:
                    sum_ratings += sim * r_normalise_neighbor
                else:
                    sum_ratings += sim * r
                actual_k += 1


        # if actual_k < self.min_k:
        #     sum_sim = sum_ratings = actual_k = 0
        #     for user, rating in self.normalized_rating[i]:
        #         r_normalise_neighbor = self.all_user_info[user]['normalized_r'][i]
        #         sum_ratings += r_normalise_neighbor
        #         actual_k += 1
        #         sum_sim += 1

            #raise PredictionImpossible('Not enough neighbors.')

        #est = self.all_user_info[u]['mean_r'] + (sum_ratings / sum_sim)*self.all_user_info[u]['std_r']
        if actual_k < self.min_k:
            est = self.trainset.global_mean + self.bu[x] + self.bi[y]
        else:
            if self.normalise_rating:
                est = self.all_user_info[x]['mean_r'] + (sum_ratings / sum_sim) * self.all_user_info[x]['std_r']
            else:
                est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class EbcrNormPccKNN(KNNBasic):

    """
    The variant of a basic collaborative filtering algorithm using pearson correlation coefficient measure adjusted by EBCR ratio.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation. Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
        alpha0(double), beta0(double): The hyper-parameters of the distribution of concordance ratios, they will be estimated
            by maximum likelihood estimator during the training phase.
        normalise_rating(bool): Whether to use normalized ratings during the prediction phase.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, threshold=0, alpha0=0.0, beta0=0.0, min_freq=0, normalise_rating=True, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                          **kwargs)
        self.k = k
        self.min_k = min_k
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.all_user_info = {}
        self.normalized_rating = {}
        self.normalise_rating = normalise_rating
        self.min_freq = min_freq
        self.threshold = threshold

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        self.bu, self.bi = self.compute_baselines()
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        freq = np.zeros((n_x, n_x), np.int)
        concordant_freq = np.zeros((n_x, n_x), np.int)
        prods = np.zeros((n_x, n_x), np.double)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        si = np.zeros((n_x, n_x), np.double)
        sj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        self.all_user_info, self.normalized_item_rating = self.normalize_user_rating(trainset)

        for item, y_ratings in iteritems(self.normalized_item_rating):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    prods[xi, xj] += ri * rj
                    freq[xi, xj] += 1
                    sqi[xi, xj] += ri ** 2
                    sqj[xi, xj] += rj ** 2
                    si[xi, xj] += ri
                    sj[xi, xj] += rj
                    if self.is_concordant(ri, rj):
                        concordant_freq[xi, xj] += 1

        concordant_ratio_list = list()
        for xi in range(n_x):
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] > self.min_freq:  # tune nb co-rated item for large dataset
                    ratio = concordant_freq[xi, xj] / freq[xi, xj]
                    if ratio == 0:
                        ratio = 0.001
                    if ratio == 1:
                        ratio = 0.999
                    concordant_ratio_list.append(ratio)
        # print(len(concordant_ratio_list))

        # if self.alpha0 == 0.0 and self.beta0 == 0.0:
        alpha0, beta0, _, __ = beta.fit(concordant_ratio_list, floc=0, fscale=1)
        self.alpha0 = alpha0
        self.beta0 = beta0

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                # norm_sq_diff = sq_diff[xi, xj] / freq[xi, xj]
                if freq[xi, xj] == 0:
                    sim[xi, xj] = 0
                else:
                    n = freq[xi, xj]
                    num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                    denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj] ** 2) *
                                    (n * sqj[xi, xj] - sj[xi, xj] ** 2))
                    if denum == 0:
                        sim[xi, xj] = 0
                    else:
                        sim[xi, xj] = ((concordant_freq[xi, xj] + self.alpha0) / (freq[xi, xj] + self.alpha0 + self.beta0)) \
                                  * ((num / denum) + 1) / 2

                sim[xj, xi] = sim[xi, xj]
        self.sim = sim

        return self

    def like_item_by_std(self, normalized_r_u, std):
        if (normalized_r_u > std):  ##  evaluate parameter (std)
            return 1
        if (normalized_r_u < -std):
            return -1
        return 0

    def is_concordant(self, normalized_r_u1, normalized_r_u2):
        return self.like_item_by_std(normalized_r_u1, 0.5) == self.like_item_by_std(normalized_r_u2, 0.5)

    def normalize_user_rating(self, trainset):
        ub = self.sim_options['user_based']
        overall_sigma = np.std([r for (_, _, r)
                                in trainset.all_ratings()])
        user_rating = trainset.ur if ub else trainset.ir
        # item_rating = dataset.yr
        all_user_info = {}
        normalized_item_rating = {}
        item_ratings = trainset.ir if ub else trainset.ur
        for item in item_ratings.keys():
            # for every item as key in a dict, construct a empty list
            normalized_item_rating[item] = list()

        for user, item_rating in iteritems(user_rating):
            user_info = {}
            user_mean = np.mean([rating for (item, rating) in item_rating])
            sigma = np.std([rating for (item, rating) in item_rating])
            user_std = overall_sigma if sigma == 0 else sigma
            user_normalise_r_value = [(item, (rating - user_mean) / user_std) for (item, rating) in item_rating]
            user_normalise_r_value_dict = {}

            for item, rating in user_normalise_r_value:
                user_normalise_r_value_dict[item] = rating

            for (item, norm_rating) in user_normalise_r_value:
                normalized_item_rating[item].append((user, norm_rating))
            user_info['mean_r'] = user_mean
            user_info['std_r'] = user_std
            user_info['normalized_r'] = user_normalise_r_value_dict
            all_user_info[user] = user_info

        return (all_user_info, normalized_item_rating)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        # k_neighbors.pop(0) #delete the neighbor who is the user itself

        # print(k_neighbors)
        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0

        for (sim, r, neighbor_inner_id) in k_neighbors:
            if sim > self.threshold:
                sum_sim += sim
                r_normalise_neighbor = self.all_user_info[neighbor_inner_id]['normalized_r'][y]
                if self.normalise_rating:
                    sum_ratings += sim * r_normalise_neighbor
                else:
                    sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            est = self.trainset.global_mean + self.bu[x] + self.bi[y]
        else:
            if self.normalise_rating:
                est = self.all_user_info[x]['mean_r'] + (sum_ratings / sum_sim) * self.all_user_info[x]['std_r']
            else:
                est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class NormPcc(KNNBasic):

    """
    The variant of a basic collaborative filtering algorithm using normalized pearson correlation coefficient measure.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                          **kwargs)
        self.k = k
        self.min_k = min_k


    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        freq = np.zeros((n_x, n_x), np.int)
        prods = np.zeros((n_x, n_x), np.double)
        union = np.zeros((n_x, n_x), np.int)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        si = np.zeros((n_x, n_x), np.double)
        sj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    prods[xi, xj] += ri * rj
                    freq[xi, xj] += 1
                    sqi[xi, xj] += ri ** 2
                    sqj[xi, xj] += rj ** 2
                    si[xi, xj] += ri
                    sj[xi, xj] += rj

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj] ** 2) *
                                (n * sqj[xi, xj] - sj[xi, xj] ** 2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = ((num / denum) + 1) / 2

                sim[xj, xi] = sim[xi, xj]

        self.sim = sim

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self


class SW_Norm_PccKNN(KNNBasic):

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                          **kwargs)
        self.k = k
        self.min_k = min_k


    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        freq = np.zeros((n_x, n_x), np.int)
        prods = np.zeros((n_x, n_x), np.double)
        union = np.zeros((n_x, n_x), np.int)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        si = np.zeros((n_x, n_x), np.double)
        sj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    prods[xi, xj] += ri * rj
                    freq[xi, xj] += 1
                    sqi[xi, xj] += ri ** 2
                    sqj[xi, xj] += rj ** 2
                    si[xi, xj] += ri
                    sj[xi, xj] += rj

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj] ** 2) *
                                (n * sqj[xi, xj] - sj[xi, xj] ** 2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = ((num / denum) + 1) / 2

                sim[xi, xj] = (n / 50) * sim[xi, xj] if n < 50 else sim[xi, xj]

                sim[xj, xi] = sim[xi, xj]

        self.sim = sim

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self


class SW_MSD_KNN(KNNBasic):

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                          **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        sq_diff = np.zeros((n_x, n_x), np.double)
        freq = np.zeros((n_x, n_x), np.int)
        sim = np.zeros((n_x, n_x), np.double)

        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    sq_diff[xi, xj] += (ri - rj)**2
                    freq[xi, xj] += 1

        for xi in range(n_x):
            sim[xi, xi] = 1  # completely arbitrary and useless anyway
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] < 1:
                    sim[xi, xj] = 0
                else:
                    # return inverse of (msd + 1) (+ 1 to avoid dividing by zero)
                    sim[xi, xj] = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)

                sim[xi, xj] = (freq[xi, xj] / 50) * sim[xi, xj] if freq[xi, xj] < 50 else sim[xi, xj]

                sim[xj, xi] = sim[xi, xj]

        self.sim = sim

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self


class SW_COS_KNN(KNNBasic):

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                          **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        prods = np.zeros((n_x, n_x), np.double)
        freq = np.zeros((n_x, n_x), np.int)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    freq[xi, xj] += 1
                    prods[xi, xj] += ri * rj
                    sqi[xi, xj] += ri**2
                    sqj[xi, xj] += rj**2

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] < 1:
                    sim[xi, xj] = 0
                else:
                    denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                    sim[xi, xj] = prods[xi, xj] / denum

                sim[xi, xj] = (freq[xi, xj] / 50) * sim[xi, xj] if freq[xi, xj] < 50 else sim[xi, xj]

                sim[xj, xi] = sim[xi, xj]

        self.sim = sim

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self


class LS_MSD_KNN(KNNBasic):
    """
    The variant of a basic collaborative filtering algorithm using mean squared distance measure adjusted by EBCR ratio.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation. Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
        alpha0(double), beta0(double): The hyper-parameters of the distribution of concordance ratios, they will be estimated
            by maximum likelihood estimator during the training phase.
        normalise_rating(bool): Whether to use normalized ratings during the prediction phase.
    """

    def __init__(self, k=40, min_k=1, a=0.5, sim_options={}, alpha0=1, beta0=1, min_freq=0, normalise_rating=True, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.all_user_info = {}
        self.normalized_rating = {}
        self.normalise_rating = normalise_rating
        self.min_freq = min_freq
        self.a = a

    def fit(self, trainset):
        """
        The fit function calculate the similarity matrix for the given train set
        """
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        self.bu, self.bi = self.compute_baselines()
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur
        # print(n_x)
        freq = np.zeros((n_x, n_x), np.int)
        concordant_freq = np.zeros((n_x, n_x), np.int)
        sq_diff = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        self.all_user_info, self.normalized_item_rating = self.normalize_user_rating(trainset)

        for item, y_ratings in iteritems(self.normalized_item_rating):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    sq_diff[xi, xj] += (ri - rj) ** 2
                    freq[xi, xj] += 1
                    if self.is_concordant(ri, rj):
                        concordant_freq[xi, xj] += 1

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                # norm_sq_diff = sq_diff[xi, xj] / freq[xi, xj]
                if freq[xi, xj] == 0:
                    sim[xi, xj] = 0
                else:
                    inverse_diff = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)
                    sim[xi, xj] = ((concordant_freq[xi, xj] + self.alpha0) / (freq[xi, xj] + self.alpha0 + self.beta0)) \
                                  * inverse_diff
                sim[xj, xi] = sim[xi, xj]
        self.sim = sim
        # print(sim)
        return self

    def like_item_by_std(self, normalized_r_u, std):
        """
        determine the user taste given the centered and reduced rating and the standard deviation
        """
        if(normalized_r_u > std):
            return 1
        if (normalized_r_u < -std):
            return -1
        return 0

    def is_concordant(self, normalized_r_u1, normalized_r_u2):
        """
        function to determine whether or not two ratings are concordant
        """
        return self.like_item_by_std(normalized_r_u1, self.a) == self.like_item_by_std(normalized_r_u2, self.a)

    def normalize_user_rating(self, trainset):
        """
        function to normalize user ratings according to their own rating systems
        """
        ub = self.sim_options['user_based']

        overall_sigma = np.std([r for (_, _, r)
                                in trainset.all_ratings()])
        user_rating = trainset.ur if ub else trainset.ir
        # item_rating = dataset.yr
        all_user_info = {}
        normalized_item_rating = {}
        item_ratings = trainset.ir if ub else trainset.ur
        for item in item_ratings.keys():
            #for every item as key in a dict, construct a empty list
            normalized_item_rating[item] = list()

        for user, item_rating in iteritems(user_rating):
            user_info = {}
            user_mean = np.mean([rating for (item, rating) in item_rating])
            sigma = np.std([rating for (item, rating) in item_rating])
            user_std = overall_sigma if sigma == 0 else sigma
            user_normalise_r_value = [(item, (rating - user_mean) / user_std) for (item, rating) in item_rating]
            user_normalise_r_value_dict = {}

            for item, rating in user_normalise_r_value:
                user_normalise_r_value_dict[item] = rating

            for (item, norm_rating) in user_normalise_r_value:
                normalized_item_rating[item].append((user, norm_rating))
            user_info['mean_r'] = user_mean
            user_info['std_r'] = user_std
            user_info['normalized_r'] = user_normalise_r_value_dict
            all_user_info[user] = user_info

        return (all_user_info, normalized_item_rating)

    def estimate(self, u, i):
        """
        function to predict the rating value of the given user and item
        """

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        sum_sim = sum_ratings = actual_k = 0

        for (sim, r, neighbor_inner_id) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                r_normalise_neighbor = self.all_user_info[neighbor_inner_id]['normalized_r'][y]
                if self.normalise_rating:
                    sum_ratings += sim * r_normalise_neighbor
                else:
                    sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            est = self.trainset.global_mean + self.bu[x] + self.bi[y]
        else:
            if self.normalise_rating:
                est = self.all_user_info[x]['mean_r'] + (sum_ratings / sum_sim) * self.all_user_info[x]['std_r']
            else:
                est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class LS_COS_KNN(KNNBasic):

    """
    The variant of a basic collaborative filtering algorithm using cosine metric adjusted by EBCR ratio.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation. Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
        alpha0(double), beta0(double): The hyper-parameters of the distribution of concordance ratios, they will be estimated
            by maximum likelihood estimator during the training phase.
        normalise_rating(bool): Whether to use normalized ratings during the prediction phase.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, alpha0=1, beta0=1, min_freq=0, concordant_only=True, normalise_rating=True, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.all_user_info = {}
        self.normalized_rating = {}
        self.concordant_only = concordant_only
        self.normalise_rating = normalise_rating
        self.min_freq = min_freq

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        self.bu, self.bi = self.compute_baselines()
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        prods = np.zeros((n_x, n_x), np.double)
        freq = np.zeros((n_x, n_x), np.int)
        concordant_freq = np.zeros((n_x, n_x), np.int)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        self.all_user_info, self.normalized_item_rating = self.normalize_user_rating(trainset)

        for item, y_ratings in iteritems(self.normalized_item_rating):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    prods[xi, xj] += ri * rj
                    sqi[xi, xj] += ri ** 2
                    sqj[xi, xj] += rj ** 2
                    freq[xi, xj] += 1
                    if self.is_concordant(ri, rj):
                        concordant_freq[xi, xj] += 1

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                # norm_sq_diff = sq_diff[xi, xj] / freq[xi, xj]
                if freq[xi, xj] == 0:
                    sim[xi, xj] = 0
                else:
                    denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                    sim[xi, xj] = ((concordant_freq[xi, xj] + self.alpha0) / (freq[xi, xj] + self.alpha0 + self.beta0)) \
                                  * (prods[xi, xj] / denum)

                sim[xj, xi] = sim[xi, xj]
        self.sim = sim

        return self

    def like_item_by_std (self, normalized_r_u, std):
        if(normalized_r_u > std): ##  evaluate parameter (std)
            return 1
        if (normalized_r_u < -std):
            return -1
        return 0

    def is_concordant(self, normalized_r_u1, normalized_r_u2):
        return self.like_item_by_std(normalized_r_u1, 0.5) == self.like_item_by_std(normalized_r_u2, 0.5)

    def normalize_user_rating(self, trainset):
        ub = self.sim_options['user_based']
        overall_sigma = np.std([r for (_, _, r)
                                in trainset.all_ratings()])
        user_rating = trainset.ur if ub else trainset.ir
        # item_rating = dataset.yr
        all_user_info = {}
        normalized_item_rating = {}
        item_ratings = trainset.ir if ub else trainset.ur
        for item in item_ratings.keys():
            #for every item as key in a dict, construct a empty list
            normalized_item_rating[item] = list()

        for user, item_rating in iteritems(user_rating):
            user_info = {}
            user_mean = np.mean([rating for (item, rating) in item_rating])
            sigma = np.std([rating for (item, rating) in item_rating])
            user_std = overall_sigma if sigma == 0 else sigma
            user_normalise_r_value = [(item, (rating - user_mean) / user_std) for (item, rating) in item_rating]
            user_normalise_r_value_dict = {}

            for item, rating in user_normalise_r_value:
                user_normalise_r_value_dict[item] = rating

            for (item, norm_rating) in user_normalise_r_value:
                normalized_item_rating[item].append((user, norm_rating))
            user_info['mean_r'] = user_mean
            user_info['std_r'] = user_std
            user_info['normalized_r'] = user_normalise_r_value_dict
            all_user_info[user] = user_info

        return (all_user_info, normalized_item_rating)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        # k_neighbors.pop(0) #delete the neighbor who is the user itself

        # print(k_neighbors)
        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0

        for (sim, r, neighbor_inner_id) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                r_normalise_neighbor = self.all_user_info[neighbor_inner_id]['normalized_r'][y]
                if self.normalise_rating:
                    sum_ratings += sim * r_normalise_neighbor
                else:
                    sum_ratings += sim * r
                actual_k += 1


        # if actual_k < self.min_k:
        #     sum_sim = sum_ratings = actual_k = 0
        #     for user, rating in self.normalized_rating[i]:
        #         r_normalise_neighbor = self.all_user_info[user]['normalized_r'][i]
        #         sum_ratings += r_normalise_neighbor
        #         actual_k += 1
        #         sum_sim += 1

            #raise PredictionImpossible('Not enough neighbors.')

        #est = self.all_user_info[u]['mean_r'] + (sum_ratings / sum_sim)*self.all_user_info[u]['std_r']
        if actual_k < self.min_k:
            est = self.trainset.global_mean + self.bu[x] + self.bi[y]
        else:
            if self.normalise_rating:
                est = self.all_user_info[x]['mean_r'] + (sum_ratings / sum_sim) * self.all_user_info[x]['std_r']
            else:
                est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class LS_Norm_PccKNN(KNNBasic):

    """
    The variant of a basic collaborative filtering algorithm using pearson correlation coefficient measure adjusted by EBCR ratio.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation. Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
        alpha0(double), beta0(double): The hyper-parameters of the distribution of concordance ratios, they will be estimated
            by maximum likelihood estimator during the training phase.
        normalise_rating(bool): Whether to use normalized ratings during the prediction phase.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, threshold=0, alpha0=1, beta0=1, min_freq=0, normalise_rating=True, verbose=True, **kwargs):
        KNNBasic.__init__(self, sim_options=sim_options, verbose=verbose,
                          **kwargs)
        self.k = k
        self.min_k = min_k
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.all_user_info = {}
        self.normalized_rating = {}
        self.normalise_rating = normalise_rating
        self.min_freq = min_freq
        self.threshold = threshold

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        self.bu, self.bi = self.compute_baselines()
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        freq = np.zeros((n_x, n_x), np.int)
        concordant_freq = np.zeros((n_x, n_x), np.int)
        prods = np.zeros((n_x, n_x), np.double)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        si = np.zeros((n_x, n_x), np.double)
        sj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        self.all_user_info, self.normalized_item_rating = self.normalize_user_rating(trainset)

        for item, y_ratings in iteritems(self.normalized_item_rating):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    prods[xi, xj] += ri * rj
                    freq[xi, xj] += 1
                    sqi[xi, xj] += ri ** 2
                    sqj[xi, xj] += rj ** 2
                    si[xi, xj] += ri
                    sj[xi, xj] += rj
                    if self.is_concordant(ri, rj):
                        concordant_freq[xi, xj] += 1


        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                # norm_sq_diff = sq_diff[xi, xj] / freq[xi, xj]
                if freq[xi, xj] == 0:
                    sim[xi, xj] = 0
                else:
                    n = freq[xi, xj]
                    num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                    denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj] ** 2) *
                                    (n * sqj[xi, xj] - sj[xi, xj] ** 2))
                    if denum == 0:
                        sim[xi, xj] = 0
                    else:
                        sim[xi, xj] = ((concordant_freq[xi, xj] + self.alpha0) / (freq[xi, xj] + self.alpha0 + self.beta0)) \
                                  * ((num / denum) + 1) / 2

                sim[xj, xi] = sim[xi, xj]
        self.sim = sim

        return self

    def like_item_by_std(self, normalized_r_u, std):
        if (normalized_r_u > std):  ##  evaluate parameter (std)
            return 1
        if (normalized_r_u < -std):
            return -1
        return 0

    def is_concordant(self, normalized_r_u1, normalized_r_u2):
        return self.like_item_by_std(normalized_r_u1, 0.5) == self.like_item_by_std(normalized_r_u2, 0.5)

    def normalize_user_rating(self, trainset):
        ub = self.sim_options['user_based']
        overall_sigma = np.std([r for (_, _, r)
                                in trainset.all_ratings()])
        user_rating = trainset.ur if ub else trainset.ir
        # item_rating = dataset.yr
        all_user_info = {}
        normalized_item_rating = {}
        item_ratings = trainset.ir if ub else trainset.ur
        for item in item_ratings.keys():
            # for every item as key in a dict, construct a empty list
            normalized_item_rating[item] = list()

        for user, item_rating in iteritems(user_rating):
            user_info = {}
            user_mean = np.mean([rating for (item, rating) in item_rating])
            sigma = np.std([rating for (item, rating) in item_rating])
            user_std = overall_sigma if sigma == 0 else sigma
            user_normalise_r_value = [(item, (rating - user_mean) / user_std) for (item, rating) in item_rating]
            user_normalise_r_value_dict = {}

            for item, rating in user_normalise_r_value:
                user_normalise_r_value_dict[item] = rating

            for (item, norm_rating) in user_normalise_r_value:
                normalized_item_rating[item].append((user, norm_rating))
            user_info['mean_r'] = user_mean
            user_info['std_r'] = user_std
            user_info['normalized_r'] = user_normalise_r_value_dict
            all_user_info[user] = user_info

        return (all_user_info, normalized_item_rating)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r, x2) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        # k_neighbors.pop(0) #delete the neighbor who is the user itself

        # print(k_neighbors)
        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0

        for (sim, r, neighbor_inner_id) in k_neighbors:
            if sim > self.threshold:
                sum_sim += sim
                r_normalise_neighbor = self.all_user_info[neighbor_inner_id]['normalized_r'][y]
                if self.normalise_rating:
                    sum_ratings += sim * r_normalise_neighbor
                else:
                    sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            est = self.trainset.global_mean + self.bu[x] + self.bi[y]
        else:
            if self.normalise_rating:
                est = self.all_user_info[x]['mean_r'] + (sum_ratings / sum_sim) * self.all_user_info[x]['std_r']
            else:
                est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details
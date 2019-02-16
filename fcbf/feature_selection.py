import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import OrderedDict
from typing import Dict, Union, List

def get_dists(X: pd.Series, Y: pd.Series):
    '''
    Discrete distribution for column X, Y
    '''

    df = pd.concat([X, Y], axis=1)
    X_dist = df.groupby(X.name).size().div(df.shape[0])
    Y_dist = df.groupby(Y.name).size().div(df.shape[0])

    return X_dist, Y_dist, df

def HXY(X: pd.Series, Y: pd.Series, base=np.e):
    '''
    H[X|Y]
    '''
    X_dist, Y_dist, df = get_dists(X, Y)

    con_probs = df.groupby([X.name, Y.name]).size().div(df.shape[0]).div(Y_dist, axis=0, level=Y.name)
    con_probs = con_probs.swaplevel()

    return np.dot(
        [Y_dist[each] for each in list(Y_dist.index)],
        [entropy(con_probs[each], base=base) for each in list(Y_dist.index)]
    )

def IG(X: pd.Series, Y: pd.Series, base=np.e):
    '''
    IG[X|Y] = H[X] - H[X|Y]
    '''
    H_X = entropy(get_dists(X, Y)[0], base=base)
    H_XY = HXY(X, Y, base=base)

    assert H_X >= 0, 'Entropy should be bigger than 0'
    assert H_X >= H_XY, 'Entropy should be bigger than its conditional entropy'

    return H_X - H_XY


def SU(X: pd.Series, Y: pd.Series, base=np.e):
    '''
    SU(X,Y) = 2 * IG[X|Y] / ( H[X] + H[Y] )
    note that IG is symmetric
    '''

    X_dist, Y_dist, _ = get_dists(X, Y)

    H_X = entropy(X_dist, base=base)
    H_Y = entropy(Y_dist, base=base)
    IG_XY = IG(X, Y, base=base)

    return 2 * IG_XY / (H_X + H_Y)

def ig_su(X: pd.Series, Y: pd.Series, base=np.e):
    return IG(X, Y, base=base), SU(X, Y, base=base)

def fcbf(X:pd.DataFrame, y:pd.Series, threshold=0.0, base=np.e, is_debug=False) -> Union[Dict[str, int], List]:
    '''

    :param X: pd.DataFrame that holds features
    :param y: pd.Series that holds target
    :param threshold: threshold for class relavance, refer to paper for details
    :param base: base for logarithm
    :param is_debug: if true, will print more details
    :return:
        1. Most relavant features to class and their degree
        2. Redundant feature(s) to each feature
    '''

    S_list= dict()

    for feature in X:
        Fi = X[feature]
        SUic = SU(X=Fi, Y=y, base=base)

        if SUic > threshold:
            S_list.update({feature: SUic})

    S_list = sorted(S_list.items(), key=lambda kv: kv[1], reverse=True)
    remove_history = dict()

    if is_debug:
        print('original features = ', S_list)

    idx_j = 0
    while idx_j < len(S_list):
        feature_j, _ = S_list[idx_j]
        if is_debug:
            print('\t', 'Fj = ', feature_j)
        Fj = X[feature_j]
        idx_i = idx_j + 1

        if idx_i < len(S_list):
            remove_fi = []
            SUij_history = []
            for i in range(idx_i, len(S_list)):
                feature_i, SUic = S_list[i]
                Fi = X[feature_i]
                SUij = SU(Fi, Fj, base=base)

                if is_debug:
                    print('\t\t', 'Fi = ', feature_i,)

                if SUij >= SUic:
                    remove_fi.append(S_list[i])
                    SUij_history.append({feature_i: SUij})
                    if is_debug:
                        print('\t\t\t (Redundant)','SUij = ',SUij, 'SUic', SUic)

            for each in remove_fi:
                S_list.remove(each)

            remove_history.update({feature_j: SUij_history})

        # If idx_j is located at the tail of S_list, then it should stop
        else:
            break

        idx_j += 1

    if is_debug:
        print('removed history for each feature', remove_history)
        print('best features = ', S_list, '\n')

    correlation_dict = sorted(
        {feature: SUic for feature, SUic in S_list}.items(),
        key=lambda kv: kv[1],
        reverse=True)

    return correlation_dict, remove_history
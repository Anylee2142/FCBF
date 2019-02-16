from fcbf.feature_selection import *

import numpy as np
import pandas as pd
import unittest

class TestEntropy(unittest.TestCase):
    def test_conditional_entropy(self):
        data = pd.read_csv('dataset/titanic/train.csv')
        X = data['Embarked']
        y = data['Survived']

        X_dist, y_dist, _ = get_dists(X, y)

        H_X = entropy(X_dist, base=2)
        H_XY = HXY(X, y, base=2)

        self.assertGreaterEqual(H_X, 0)
        self.assertGreaterEqual(H_XY, 0)
        self.assertGreaterEqual(H_X, H_XY)

        info_gain = IG(X, y, base=2)
        self.assertGreaterEqual(info_gain, 0)

        symmetric_uncertainty = SU(X, y, base=2)
        self.assertGreaterEqual(symmetric_uncertainty, 0)

if __name__ == '__main__':
    unittest.main()
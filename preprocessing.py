# preprocessing.py

"""
Functions for data cleaning, imputation, feature mapping and scaling.
"""
import numpy as np
import pandas as pd
import sys

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import TransformerMixin


__all__ = ['yn_binarize',
           'map_feature',
           'impute_by_other',
           'impute_all',
           'DataFrameImputer',
           'strat_shufflesplit',
           'random_oversample',
           'random_undersample',
           'balance_classes_smote',
           'balance_classes_adasyn',
           'PolynomialConstructor']


# ===== Feature Mapping
def yn_binarize(df, features):
    """
    Function to all specified 'yes'/'no' features yes=1, no=2. Note, if data is missing for a given feature, it should
        be cleaned or imputed prior to the use of this function. Otherwise, the missing value may not be mapped.
    :param df: pandas DataFrame with feature data
    :param features: (string) List of feature column names in the DataFrame that have only 'yes'/'no' values.
    :return:
    """
    bmap = {'yes': 1, 'no': 0}
    for feature in features:
        unique = np.unique(df[feature].values)
        print('Unqiue values for feature %s: %s' % (feature, unique))
        print('Mapping %s to yes=1, no=0', unique)
        df[feature] = df[feature].map(bmap)
    return df


def map_feature(df, feature, feature_values, mapped_values):
    """
    Function to map a categorical feature within a pandas DataFrame to the values specified in
     parameter 'mapped_values' in a manner corresponding to the order/content of parameter 'feature_values'
    :param df: (pandas DataFrame) DataFrame containing the the features to be mapped
    :param feature: (string) The column name of the feature to be mapped
    :param feature_values: (string) List of values (string expected) to be mapped to new values as specified by the
        parameter mapped_values.
    :param mapped_values: (any) A list of values to map the list of possible feature values in parameter feature_values.
        Note, feature_values and mapped_values must be of equal length.
    :return:
    """
    if feature not in df.columns.values:
        print('In function "map_feature", feature %s not found in DataFrame!' % feature)
        return None
    elif len(feature_values) != len(mapped_values):
        print('In function "map_feature", parameters feature_values and mapped_values must be'
              'of equal length for dictionary creation prior to mapping.')
        return None
    else:
        map_dict = dict(zip(feature_values, mapped_values))
        df[feature] = df[feature].map(map_dict)
    return df


# ====== Imputation
def impute_by_other(df, imp_feat, other_feat):
    """
    Function to impute a feature (imp_feat) by the corresponding mean of another features value in that sample.
    The average is calculated from other samples where the imputed feature is not null and the value of the other
    feature (other_feat) is the same as for the sample being imputed.
    :param df: pandas DataFrame containing the feature to be imputed as well as the reference feature.
    :param other_feat: Column name of the other feature from which to take the mean of imp_feat
    :param imp_feat: Column name of the feature to be imputed with the mean derived from the other feature.
    :return: None (original DataFrame modified)
    """
    df_clean = df[[other_feat, imp_feat]].dropna(axis=0, how='any')

    for other in np.unique(df[other_feat].values):
        avg_feature = df_clean[df_clean[other_feat] == other][imp_feat].mean()
        df.ix[(df[other_feat] == other) & (df[imp_feat].isnull()), imp_feat] = avg_feature
    return None


def impute_all(df):
    """
    Function to impute all nan values in a pandas DataFrame using the class DataFrameImputer as defined with
    this modeule. Numeric features are imputed as the mean of the column wheras object/string values are imputed
    by the median. The same result can be achieved by using the .transform method of the DataFrameImputer class, and
    this function basically acts as a wrapper.
    :param df: (pandas df) DataFrame containing the feature data that will be imputed.
    :return:
    """
    df_imp = DataFrameImputer()
    return df_imp.fit_transform(df)


class DataFrameImputer(TransformerMixin):
    """
    Class for the imputation of features either numerical or object. The methods transform and fit_transform
    return either a) objects by the most frequent value in a column or b) the mean numerical value in a column.
    If y is passed, it is not affected.

    Credit for this code goes to Stackoverflow user sveitser.
    """
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# ====== Data Splitting and Class Balancing (up-sampling/down-sampling)
def strat_shufflesplit(X, y, nsplits=10, test_size=0.2, rand=0):
    """
    Function to perform stratified shuffle split (Sci-kit Learn implementation) to create test/train data
        sets with equal numbers of classes.
    :param X: Matrix of features
    :param y: Class labels
    :param nsplits: Number of splits to pass to the n_splits parameter for StratifiedShuffleSploit
    :param test_size: Fraction of the data to put into the train/test set.
    :param rand: (int): Random state seed
    :return:
    """

    sss = StratifiedShuffleSplit(n_splits=nsplits, test_size=test_size, random_state=rand)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
        else:
            print('ERROR: StratifiedShuffleSplit Failed! Both X and y must be of the same type (either numpy.ndarry '
                  'or pandas.DataFrame')
            return None
        print('Data set split using StratifiedShuffleSplit. Shapes of returned arrays:\n'
              'X_train:     %s\n'
              'X_test:      %s\n'
              'y_train:     %s\n'
              'y_test:      %s' % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))
        return X_train, X_test, y_train, y_test


def random_oversample(X, y, ratio='auto', random_state=None):
    """
    Function to oversample minority class by sampling at random with replacement (by default)
    :param X: Feature data
    :param y: Class labels
    :param ratio: (string/float) The number of samples in the minority class over the number of samples
        in the majority class.
    :param random_state: (int) Seed used by the random number generator.
    :return: Re-sampled features and corresponding class labels, with higher sampling of minority class
    """
    ros = RandomOverSampler(ratio=ratio,
                            random_state=random_state)
    X_res, y_res = ros.fit_sample(X, y)

    return X_res, y_res


def random_undersample(X, y, ratio='auto', random_state=None, replacement=True):
    """
    Function to undersample majority class by sampling at random with replacement (by default)
    :param X: Feature data
    :param y: Class labels
    :param ratio: (string/float) The number of samples in the minority class over the number of samples
        in the majority class.
    :param random_state: (int) Seed used by the random number generator.
    :param replacement: (bool) Whether or not to sample with replacement
    :return: Re-sampled features and corresponding class labels, with lower sampling of majority class
    """
    rus = RandomUnderSampler(ratio=ratio,
                             random_state=random_state,
                             replacement=replacement)
    X_res, y_res = rus.fit_sample(X, y)

    return X_res, y_res


def balance_classes_smote(X, y, kind='regular', ratio='auto', m=5, k=15):
    """
    Function to balance the distribute of classes by using Synthetic Minority
    Over-sampling Technique (SMOTE)
    :param X: Feature data
    :param y: Class labels
    :param kind: (str) SMOTE method to use, as specified by the 'kind' parameter in imblearn.over_sampling.SMOTE
    :param ratio: (str) If ‘auto’, the ratio will be defined automatically to balance the dataset. Otherwise, the ratio
        is defined as the number of samples in the minority class over the the number of samples in the majority class.
    :param m: (int) Number of nearest neighbors to use to determine if a minority sample is in danger
    :param k: (int) Number of nearest neighbors used to construct synthetic samples
    :return: Data set with synthetic samples added
    """
    sm = SMOTE(kind=kind, ratio=ratio, m=m, k=k)
    X_smote, y_smote = sm.fit_sample(X, y)

    return X_smote, y_smote


def balance_classes_adasyn(X, y, ratio='auto', random_state=None, k=5):
    """
    Function to balance the distribute of classes by using Adaptive Synthetic
    Sampling Approach for Imbalanced Learning (ADASYN)
    :param X: Feature data
    :param y: Class labels
    :param ratio: (str/float) If ‘auto’, the ratio will be defined automatically to balance the dataset. Otherwise, the
        ratio is defined as the number of samples in the minority class over the the number of samples in the majority
        class.
    :param random_state: (None/Int) If int, seed used for random number generator
    :param k: (int) Number of nearest neighbors used to construct synthetic samples
    :return: Data set with synthetic samples added
    """
    ad = ADASYN(ratio=ratio,
                random_state=random_state,
                n_jobs=1,
                k=k)
    X_adasyn, y_adasyn = ad.fit_sample(X, y)

    return X_adasyn, y_adasyn


# ======= Feature engineering/generation
class PolynomialConstructor(object):
    """
    Class containing several methods for the construction of new arrays via the transformation
        existing arrays that are passed. 'fit' methods should be used if the transformation of the
        data is desired without immediate return, and transformed data will be stored in the attributes
        'X.new' or 'df.new', depending on whether the input matrix is a ndarray or pandas DataFrame.
        'fit_transform' methods should be used to return return the transformed data.

    Transformations available:
            1. poly: Calculates the nth power of the input values. E.g., x1 --> x1^n
            2. polysum: Calculates the column-pair-wise sum of values raised to the nth power.
                E.g., (x1, x2) ---> (x1^n + x2^n)
            3. polyratio: Calculates the pairwise ratios of input values after raising them to the nth power.
                E.g., (x1, x2) ---> (x1^n/x2^n, x2^n/x1^n)
            4. polyprod: Calculates the pairwise product of the input values to the nth power.
                E.g., (x1, x2) ---> (x1^n*x2^n)

    Combinations of columns in the input matrix may be transformed by any of the above methods. Original
        data is preserved and returned along with the transformed data.

    NOTE: Passing data as a pandas DataFrame is recommended so that the column names are easily
        preserved, and the components of new columns can be identified by their corresponding column titles
        in the new returned DataFrame.
    """
    def __init__(self, name='PolynomialConstructor', n=2, verbose=1):
        """
        Attributes:
            1. self.name: Customizable text name of the PolynomialConstructor object
            2. self.n: (int) Power by which to raise the values in each column of the input matrix. E.g., if n == 2,
                x --> x^2
            3. self.X_new: Placeholder for new data matrix
            4. self.df_new: Placeholder for new data DataFrame
            5. self._verbose: (int) Level of verbosity

        Common arguments:
            1. :param X: 2D matrix-like data set as ndarray or pandas DataFrame
            2. :param y: (default=None). Optional class label array. No operations are performed on y.
        """
        self.name = name
        self.n = int(n)
        self.X_new = None
        self.df_new = pd.DataFrame()
        self._verbose = verbose

    def __reset__(self):
        """
        Method to reset data from previous fit
        :return: None
        """
        self.X_new = None
        self.df_new = pd.DataFrame()

    @staticmethod
    def __err__():
        """
        Method to raise error flag when input is not ndarray or pandas DataFrame
        """
        print("ERROR: Input data must be ndarray or pandas DataFrame (recommended) format.")
        sys.exit('Quitting...')

    def fit_poly(self, X, y=None):
        """
        Method Generates k new data columns, where the values of each original column have been raised to the power of n
            and k is the original number of columns.
        :return: None
        """
        self.__reset__()

        if self._verbose >= 1:
            self._header(funcname=self.fit_poly.__name__)
        print('inside polynomial fit_poly ')
        if isinstance(X, pd.DataFrame):
            columns = X.columns.values

            for col in columns:
                self.df_new[col] = X[col]

            for col in columns:
                new_name = col + '^%s' % self.n
                self.df_new[new_name] = X[col]**self.n
            if self._verbose >= 1:
                self._fit_exit(newshape=self.df_new.shape, n_old=len(columns), dtype=type(self.df_new))
        elif isinstance(X, np.ndarray):
            n_columns = X.shape[1]
            self.X_new = np.zeros((X.shape[0], 2*n_columns))

            current_column = 0
            for col in range(n_columns):
                self.X_new[:, col] = X[:, col]
                current_column += 1

            for col in range(n_columns):
                self.X_new[:, current_column] = X[:, col]**self.n
                current_column += 1
            if self._verbose >= 1:
                self._fit_exit(newshape=self.X_new.shape, n_old=n_columns, dtype=type(self.X_new))
        else:
            self.__err__()

        return self

    def fit_transform_poly(self, X, y=None):
        """
        Method to fit and return the data transformed by poly_fit
        :return: Transformed and original data
        """
        self.fit_poly(X)
        if isinstance(X, pd.DataFrame):
            return self.df_new
        elif isinstance(X, np.ndarray):
            return self.X_new

    def fit_polysum(self, X, y=None):
        """
        Generates k(k+1)/2 new features that are linear combinations of the nth power of each feature couple, where k is
            the original number of features passed, and n is an integer.
        :return: None
        """
        self.__reset__()

        if self._verbose >= 1:
                self._header(funcname=self.fit_polysum.__name__)

        if isinstance(X, pd.DataFrame):
            features = X.columns.values

            for feature in features:
                self.df_new[feature] = X[feature]

            seen = set()
            for feature in features:
                for feature2 in features:
                    pair = tuple(sorted([feature, feature2]))
                    if pair not in seen:
                        new_feature_name = feature + '^%s' % self.n + '+' + feature2 + '^%s' % self.n
                        self.df_new[new_feature_name] = X[feature] ** self.n + X[feature2] ** self.n
                        seen.add(pair)
            if self._verbose >= 1:
                self._fit_exit(newshape=self.df_new.shape, n_old=len(features), dtype=type(self.df_new))
        elif isinstance(X, np.ndarray):
            n_features = X.shape[1]
            current_column = 0
            self.X_new = np.zeros((X.shape[0], (n_features*(n_features+1))/2 + n_features))

            for i in range(n_features):
                self.X_new[:, current_column] = X[:, i]
                current_column += 1

            seen = set()
            for i in range(n_features):
                for j in range(n_features):
                    pair = tuple(sorted([i, j]))
                    if pair not in seen:
                        self.X_new[:, current_column] = X[:, i]**self.n + X[:, j]**self.n
                        current_column += 1
                        seen.add(pair)
            if self._verbose >= 1:
                self._fit_exit(newshape=self.X_new.shape, n_old=n_features, dtype=type(self.X_new))
        else:
            self.__err__()

        return self

    def fit_transform_polysum(self, X, y=None):
        """
        Method to fit and return the data transformed by polysum_fit.
        :return: Transformed and original features
        """
        self.fit_polysum(X)
        if isinstance(X, pd.DataFrame):
            return self.df_new
        elif isinstance(X, np.ndarray):
            return self.X_new

    def fit_polyratio(self, X, y=None):
        """
        Creates k*(k-1) new features that are the ratio the nth power of column-pair-wise combinations of original data,
            and k is the original number of features. Avoids a features division by self since the result is 1.
        :return: None
        """
        self.__reset__()

        if self._verbose >= 1:
            self._header(funcname=self.fit_polyratio.__name__)

        if isinstance(X, pd.DataFrame):
            features = X.columns.values

            for feature in features:
                self.df_new[feature] = X[feature]

            for feature in features:
                for feature2 in features:
                    if not feature == feature2:
                        new_feature_name = feature + '^%s' % self.n + '/' + feature2 + '^%s' % self.n
                        self.df_new[new_feature_name] = (X[feature]**self.n) / (X[feature2]**self.n)
            if self._verbose >= 1:
                self._fit_exit(newshape=self.df_new.shape, n_old=len(features), dtype=type(self.df_new))
        elif isinstance(X, np.ndarray):
            n_features = X.shape[1]
            current_column = 0
            self.X_new = np.zeros((X.shape[0], n_features*(n_features - 1) + n_features))

            for i in range(n_features):
                self.X_new[:, current_column] = X[:, i]
                current_column += 1

            for i in range(n_features):
                for j in range(n_features):
                    if not i == j:
                        self.X_new[:, current_column] = (X[:, i]**self.n) / (X[:, j]**self.n)
                        current_column += 1
            if self._verbose >= 1:
                self._fit_exit(newshape=self.X_new.shape, n_old=n_features, dtype=type(self.X_new))
        else:
            self.__err__()

        return self

    def fit_transform_polyratio(self, X, y=None):
        """
        Calls fit_polyratio method and returns transformed data.
        :return: Original and transformed features.
        """
        self.fit_polyratio(X)
        if isinstance(X, pd.DataFrame):
            return self.df_new
        elif isinstance(X, np.ndarray):
            return self.X_new

    def fit_polyprod(self, X, y=None):
        """
        Creates k*(k+1)/2 new features that are the pair-wise product of the nth power of each input feature and
            k is the original number of features.
        :return: None
        """
        self.__reset__()

        if self._verbose >= 1:
            self._header(funcname=self.fit_polyprod.__name__)

        if isinstance(X, pd.DataFrame):
            features = X.columns.values

            for feature in features:
                self.df_new[feature] = X[feature]

            seen = set()
            for feature in features:
                for feature2 in features:
                    pair = tuple(sorted([feature, feature2]))
                    if pair not in seen:
                        new_feature_name = feature + '^%s' % self.n + '*' + feature2 + '^%s' % self.n
                        self.df_new[new_feature_name] = (X[feature]**self.n) * (X[feature2]**self.n)
                        seen.add(pair)
            if self._verbose >= 1:
                self._fit_exit(newshape=self.df_new.shape, n_old=len(features), dtype=type(self.df_new))
        elif isinstance(X, np.ndarray):
            n_features = X.shape[1]
            current_column = 0
            self.X_new = np.zeros((X.shape[0], (n_features*(n_features + 1)/2) + n_features))

            for i in range(n_features):
                self.X_new[:, current_column] = X[:, i]
                current_column += 1

            seen = set()
            for i in range(n_features):
                for j in range(n_features):
                    pair = tuple(sorted([i, j]))
                    if pair not in seen:
                        self.X_new[:, current_column] = (X[:, i]**self.n) * (X[:, j]**self.n)
                        current_column += 1
                        seen.add(pair)
            if self._verbose >= 1:
                self._fit_exit(newshape=self.X_new.shape, n_old=n_features, dtype=type(self.X_new))
        else:
            self.__err__()

        return self

    def fit_transform_polyprod(self, X, y=None):
        """
        Calls fit_polyprod and returns resulting data.
        :return: Original and transformed features
        """
        self.fit_polyprod(X)
        if isinstance(X, pd.DataFrame):
            return self.df_new
        elif isinstance(X, np.ndarray):
            return self.X_new

    def _fit_exit(self, newshape, n_old, dtype):
        print('===> %15s %i' % ('# New Features  |', (newshape[1] - n_old)))
        print('===> %15s %i (shape: (%s,%s))' % ('Total Features  |', newshape[1],
                                                 newshape[0], newshape[1]))
        print('DONE: New features stored in PolynomialConstructor instance "%s" obj.X_new or obj.df_new, depending '
              'on the original data type.\n' %
              self.name)
        print('Transformed features returned as:', dtype)

    def _header(self, funcname):
        print('\n==================================================================')
        print('%s: Generating new features using method: %s' % (self.name, funcname))
        print('==================================================================')

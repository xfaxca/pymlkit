# preprocessing.py

"""
Functions for data cleaning, imputation, feature mapping and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


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
    :return:
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    import pandas as pd
    import numpy as np

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


def random_oversample(X, y, ratio='auto', random_state=None, replacement=True):
    """
    Function to oversample minority class by sampling at random with replacement (by default)
    :param X: Feature data
    :param y: Class labels
    :param ratio: (string/float) The number of samples in the minority class over the number of samples
        in the majority class.
    :param random_state: (int) Seed used by the random number generator.
    :param replacement: (bool) Whether or not to sample with replacement
    :return: Re-sampled features and corresponding class labels, with higher sampling of minority class
    """
    from imblearn.over_sampling import RandomOverSampler
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
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(ratio=ratio,
                             random_state=random_state,
                             replacement=True)
    X_res, y_res = rus.fit_sample(X, y)

    return X_res, y_res


def balance_classes_smote(X, y, kind='regular', ratio='auto', m=5, k=15):
    """
    Function to balance the distribute of classes by using Synthetic Minority
    Over-sampling Technique (SMOTE)
    :param X: Feature data
    :param y: Class labels
    :param k: (int) Number of nearest neighbors used to construct synthetic samples
    :param kind: (str) SMOTE method to use, as specified by the 'kind' parameter in imblearn.over_sampling.SMOTE
    :return: Data set with synthetic samples added
    """
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(kind=kind, ratio=ratio, m=5, k=15)
    X_smote, y_smote = sm.fit_sample(X, y)

    return X_smote, y_smote


def balance_classes_adasyn(X, y, ratio='auto', random_state=None, k=5):
    """
    Function to balance the distribute of classes by using Adaptive Synthetic
    Sampling Approach for Imbalanced Learning (ADASYN)
    :param X: Feature data
    :param y: Class labels
    :param k: (int) Number of nearest neighbors used to construct synthetic samples
    :param ratio: (str/float) If ‘auto’, the ratio will be defined automatically to balance the dataset. Otherwise, the ratio is
        defined as the number of samples in the minority class over the the number of samples in the majority class.
    :return: Data set with synthetic samples added
    """
    from imblearn.over_sampling import ADASYN
    ad = ADASYN(ratio=ratio,
                random_state=random_state,
                n_jobs=1,
                k=5)
    X_adasyn, y_adasyn = ad.fit_sample(X, y)

    return X_adasyn, y_adasyn


# ====== Feature Engineering/Generation
class FeatureGenerator(object):
    """
    Class containing several methods for the generation of new features via the transformation
        existing features that are specified by the user. Each feature used as input will be combined
        in a specific manner (depending on the method used) with each other feature and itself, generating
        n^2 new features in addition to the n features passed.

        Note, passing data as a pandas DataFrame is recommended so that the feature names are preserved,
            and the components of new features can be identified by their corresponding column titles in the
            new returned DataFrame.
    """
    def __init__(self, name='FeatureGenerator', verbose=1):
        self.name = name
        self._version = '0.1'
        self.verbose = verbose
        # add more attributes as needed

    def fit_sq(self, X, y=None):
        """
        Generates n^2 new features that are combinations of the square of each feature couple.
        :param X: Feature matrix as ndarray or pandas DataFrame.
        :param y: (default=None). Optional class label array. No operations are performed on y.
        :return: None unless transform_sq is called. Transforms features using Sum of Squares
            and stores in object attribute if just fit_sq is called.
        """
        if self.verbose >= 1:
            # TODO: replace w/ decorator
            print('\n============================================================')
            print('FeatureGenerator instance "%s" invoked. Generating new features using Sum of Squares. Number of new '
                  'features generated from this method = n^2, where n is the # of original features. Note, original '
                  'features are preserved.' % self.name)
            print('============================================================')
        print('>Method: Generating New Features: Sum of Squares --> [(x1^2 + x1^2), (x1^2 + x2^2),...((x1^2 + xn^2)].')
        if isinstance(X, pd.DataFrame):
            print('Features passed in dataframe --> ', list(X.columns.values))
            print('\nFeature data was passed as a pandas DataFrame. Returning as same.')
            features = X.columns.values
            print('===> %15s %i' % ('# Old Features  |', len(features)))
            self.df_new = pd.DataFrame()
            new = []
            for feature in features:
                self.df_new[feature] = X[feature]
                for feature2 in features:
                    new_feature_name = feature + '^2' + '+' + feature2 + '^2'
                    new_feature_tmp = np.zeros((X.shape[0], 1))
                    # print('temp name:', new_feature_name)
                    for i in range(X.shape[0]):
                        new_feature_tmp[i, 0] = X.ix[i, feature]**2 + X.ix[i, feature2]**2
                    self.df_new[new_feature_name] = new_feature_tmp
            print('===> %15s %i' % ('# New Features  |', (self.df_new.shape[0] - len(features))))
            print('===> %15s %i (shape: (%s,%s)' % ('Total Features  |', self.df_new.shape[0],
                                               self.df_new.shape[0], self.df_new.shape[1]))
            print('>Output: New features stored in FeatureGenerator instance "%s" attribute obj.df_new.\n' %
                  self.name)
        elif isinstance(X, np.ndarray):
            # If is an np.ndarray, perform operations directly on the ndarray's. This should give the same
            # output as when using pandas (has been tested)
            n_features = X.shape[1]
            print('\nFeature data passed as np.ndarray. Returning in same format.')
            print('===> %15s %i' % ('# Old Features  |', n_features))
            self.X_new = np.zeros((X.shape[0], (n_features + (n_features**2))))
            current_column = 0
            for i in range(n_features):
                self.X_new[:, current_column] = X[:, i]
                current_column += 1
                for j in range(n_features):
                    new_feature_tmp = np.zeros((X.shape[0], 1))
                    for k in range(X.shape[0]):
                        new_feature_tmp[k, 0] = X[k, i]**2 + X[k, j]**2
                    self.X_new[:, current_column] = new_feature_tmp[:, 0]
                    current_column += 1
            print('===> %15s %i' % ('# New Features  |', (self.df_new.shape[0] - n_features)))
            print('===> %15s %i (shape: (%s,%s))' % ('Total Features  |', self.X_new.shape[0],
                                                self.X_new.shape[0], self.X_new.shape[1]))
            print('>Output: New features stored in FeatureGenerator instance "%s" attribute obj.X_new.\n' %
                  self.name)
            pass
        else:
            print('ERROR: Input features must be either a ndarray or pandas DataFrame format.')

    def transform_sq(self, X, y=None):
        """
        Method to fit and return the transformed data to user.
        :param X: Feature matrix as ndarray or pandas DataFrame.
        :param y: (default=None). Optional class label array. No operations are performed on y.
        :return: Transformed features using Sum of Squares. Total # of features returned will = n^2 + n,
            preserving the original features.
        """
        if isinstance(X, pd.DataFrame):
            self.fit_sq(X)
            return self.df_new
        elif isinstance(X, np.ndarray):
            pass
            self.fit_sq(X)
            return self.X_new
        else:
            print('ERROR: Input features must be either a ndarray or pandas DataFrame format.')
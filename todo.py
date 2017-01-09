# TODO items for specific modules
# #todo: Running requirements list
requirements = ['sklearn', 'mlxtend', 'numpy', 'pandas', 'seaborn', 'matplotlib', 'nltk', 'imblearn',
                'bs4']
# TODO: STart here: Add functions from the files below

"""
# Files to get functions from:

1. leafs.py
2. useful_code_fns.py
3. data_clean.py
4. scog_eval_V2.py (in Spark Cognition interview folder on desktop)
5. pygaero plotting functions --> (silhouette scoring, plotting, [make elbow method plot fn])
"""



# todo: general structure
# pymlkit: General tools for building machine learning models in Python, (based primarily on sklearn and mlextend)
# ---> explore: Data exploration, plotting, cleaning, class proportion discovery, etc.
# ---> model_select: Functions to aid in choosing the best model for a problem. Functions include scanning several
#                    standard models, grid searches and
# ---> preprocessing: Scaling, imputing, searching for/cleaning NaNs, mapping ordinal features, getting dummy variables)
# ---> model_eval: Accuracy assessment/statistics for models
# ---> nlp: NLP functions for cleaning text in different ways (newest package...will need to add more as I learn more)
# ---> premade: Premade models (stacking, bagging, majority vote)
            # --> before uploading, need to
"""
INDIVIDUAL MODULE TODOS:
A. preprocessing:
#   -1. Automated feature engineering using a selected # of features. Take input features and create sum of squares,
#       ratios, differences, x1*x2, sqrt(x1)*x2, etc. and return these new features. Start w/ making it for one of
#       these operations. Could perhaps do one function for each, or (BETTER IDEA) - MAKE A CLASS that has methods
#       to do all of these feature generation things.
#   0. Other class-balancing features from imblearn. (random upsampling, etc)
#   1. jack-knife functions to organize data that is imported in different formats (can also put this in explore.py
#             OR combine these two modules
#   1.25 Feature engineering functions
#             - example: taking the log of a specified feature(s) and append to the array (for both ndarray and dfs)
#             - consider other transformations like FFT, or exponential combinations
#   1.5 Scaling functions
#   3. Dimensionality reduction (PCA and LDA functions)
#   4. Feature selections (RFE, SelectKBest, etc.)  # TODO: start here
#   5. Feature importances using random forest
B. explore: NONE [perhaps merge this with preprocessing at some point]
C. nlp: # TODO:
#   1. Function to strip punctuation from a string
#   2. Function to find/count occurrence of a word within a dataset (or pattern using regex)
#   3. Look at the NLP tutorials/ipython notebooks that I did to get some ideas of ML functions to add for NLP
D. model_eval:
#           1. confusion matrix plots
#           2. Nice, printed figures of classification tables
#           3. Silhouette scores from clustering
#           4. ROC-AUC curve analysis/plotting
#           5. Log-loss plots from multiple classifiers
#           6. Decision tree plots
E. model_select:
#   0. Model scan (make one that can take a custom list and also one that does common base algs (do regressor version too)
#   2. Function to make preset bagged sets of models to see how they compare
#   2.5 Function to make some boosting models (see what meta estimators sklearn has to play around with)
#   3. Analogous functions to above using majority vote (read up to see if there is a huge dif)
#   4. Look into making functions to 'stack' these models, whatever that means. See what sklearn has
#   5. function to build dummy classifiers/regressors (see DummyClassifier class from sklearn) to compare the model
#       - vs different defaults (random guessing, guessing only one class, etc.)
#   6. Grid search functions for specific, commonly-used models
#   7. Model scan function would go in
#   8. Feature selection algorithms
#   9. Dimensionality reduction algorithms
#   10. Basic metrics functions/cross-val for classifiers and regressors
#   11. Build some default pipelines (with/without PCA, scaling, etc) - may be tricky since so many possible combos
#       - exist
F. Premade:
#   1. Premade majority vote classifier
#   2. construction of a stacking classifier using mlxtend




"""

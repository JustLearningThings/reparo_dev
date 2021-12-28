'''
Created with love by Sigmoid
@Author - Name Surname - email
'''

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from .errors import NonNumericDataError

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''
        Initialize the algorithm.
        '''

        pass

    def fit(self, X: 'np.array', y: 'np.array'=None, **fit_params):
        """
        :param X: array-like of shape (n_samples, n_features)
            input samples
        :param y: array-like of shape (n_samples,)
            target values (None for unsupervised transformations)
        :param fit_params: dict
            additional fit parameters
        :return: self
        """

        return self

    def transform(self, X: 'np.array', y: 'np.array'=None, **fit_params):
        """

        :param X: array-like of shape (n_samples, n_features)
            input samples
        :param y: array-like of shape (n_samples,)
            target values (None for unsupervised transformations)
        :param fit_params: dict
            additional fit parameters
        :return: ndarray array of shape (n_samples, n_features_new)
            transformed array
        """

        # check if data is numeric
        if not np.issubdtype(X.dtype, np.number):
            raise NonNumericDataError('The given array contains non-numeric values !')

        X_new = X.copy()

        # ...

        return X_new

    def fit_transform(self, X: 'np.array', y: 'np.array'=None, **fit_params):
        """
        Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
        :param X: array-like of shape (n_samples, n_features)
            input samples
        :param y: array-like of shape (n_samples,)
            target values (None for unsupervised transformations)
        :param fit_params: dict
            additional fit parameters
        :return: ndarray array of shape (n_samples, n_features_new)
            transformed array
        """

        return self.fit(X).transform(X)

    def apply(self, df: 'pandas DataFrame', columns: 'list'):
        """
        Apply the algorithm on a DataFrame
        :param df: pandas DataFrame
            the DataFrame with the possible NaN-values that should be imputed
        :param columns: list of strings
            the column that should be taken into account during the NaN-values imputation process
        :return:
        """

        # check if the given columns are in the DataFrame
        cols_not_in_df = [col for col in columns if col not in df]
        if cols_not_in_df:
            raise ValueError(f"{', '.join(cols_not_in_df)} are not present in the DataFrame")

        sub_df = df[columns].to_numpy()

        # check if data is numeric
        for col in sub_df:
            if not pd.api.types.is_numeric_dtype(sub_df[col]):
                raise NonNumericDataError('The given DataFrame contains non-numeric values !')

        # apply the transformation
        sub_df_new = self.fit_transform(sub_df)

        # save the transformation in the DataFrame (asta nu-s sigur daca lucreaza, trebuie de testat !)
        df.update(pd.DataFrame(sub_df_new, columns=columns))

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.proprocessing import LabelEncoder


class DFColumnSelector(BaseEstimator, TransformerMixin):
    """A transformer to select only the given columns from the input data.

    This class only handles pandas.DataFrame objects.

    Parameters
    ----------
    columns : list, required
        The list of columns to select from the input data
    inverse : bool, optional
        Return the inverse operation, i.e. return data with all columns
        except for those given
    """
    def __init__(self, columns, inverse=False):
        self.columns = columns
        self.inverse = inverse

    def fit(self, x, *args, **kwargs):
        return self

    def transform(self, x, *args, **kwargs):
        if self.inverse:
            # drop the columns
            return x.drop(self.columns, axis=1)
        else:
            # select the columns
            return x[self.columns]


class PipelineLabelEncoder(LabelEncoder):
    """Wrapper around LabelEncoder to allow it to work within a Pipeline.

    This class should be a drop-in replacement for LabelEncoder,
    but should also work as part of a sklearn Pipeline.
    """
    def fit(self, y, *args, **kwargs):
        return super(PipelineLabelEncoder, self).fit(y)

    def transform(self, y, *args, **kwargs):
        return super(PipelineLabelEncoder, self).transform(y)

    def fit_transform(self, y, *args, **kwargs):
        return super(PipelineLabelEncoder, self).fit_transform(y)


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """A transformer to convert input data to a pandas.Dataframe
    """
    def fit(self, x, *args, **kwargs):
        return self

    def transform(self, x, *args, **kwargs):
        return pd.DataFrame(x)


class DFDummyEncoder(BaseEstimator, TransformerMixin):
    """Dummy (aka one-hot) encoding of columns.
    """
    def __init__(self, columns=None, drop_first=None, prefix=None):
        self.columns = columns
        self.drop_first = drop_first
        self.prefix = prefix

    def fit(self, x, *args, **kwargs):
        return self

    def transform(self, x, *args, **kwargs):
        if self.columns is None:
            self.columns = x.columns
        return pd.concat([x.drop(self.columns, axis=1),
                          pd.get_dummies(x[self.columns],
                                         prefix=self.prefix,
                                         drop_first=self.drop_first,
                                         columns=self.columns)],
                         axis=1)

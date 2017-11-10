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
    """
    def fit(self, y, *args, **kwargs):
        return super(PipelineLabelEncoder, self).fit(y)

    def transform(self, y, *args, **kwargs):
        return super(PipelineLabelEncoder, self).transform(y)

    def fit_transform(self, y, *args, **kwargs):
        return super(PipelineLabelEncoder, self).fit_transform(y)

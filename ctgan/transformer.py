import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import ignore_warnings
from itertools import combinations

class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.

    Args:
        n_cluster (int):
            Number of modes.
        epsilon (float):
            Epsilon value.
    """

    def __init__(self, n_clusters=10, epsilon=0.005):
        self.n_clusters = n_clusters
        self.epsilon = epsilon

    @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, column, data, dim=1):
        gm = BayesianGaussianMixture(
            self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        gm.fit(data)
        components = gm.weights_ > self.epsilon
        num_components = components.sum()

        return {
            'name': column,
            'model': gm,
            'components': components,
            'output_info': [(dim, 'tanh'), (num_components, 'softmax')], # dim = 1 or 2
            'output_dimensions': dim + num_components,
        }

    def _fit_discrete(self, column, data):
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(data)
        categories = len(ohe.categories_[0])

        return {
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    def fit(self, data, discrete_columns=tuple()):
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        self.meta = []
        
        # since we will use 2D data to perform on Bayesian Gaussian Mixture fit,
        # we first see correlations in each continuous column, and combine two columns
        # with the highest correlation.
        columns_continuous = set(data.columns.values) - set(discrete_columns) # continous columns
        dic_ = {}
        for col1, col2 in combinations(columns_continuous, 2):
            tmp_corr = data[col1].corr(data[col2])
            if abs(tmp_corr) > 0.1:
                dic_[(col1, col2)] = tmp_corr
        # now we finish selecing all twin-columns with abs(corr)>0.1
        # print (dic_)
        dic_comb = [] # final twin-column list
        for key, _ in sorted(dic_.items(), key=lambda item: item[1], reverse=True):
            if len(dic_comb)==0 or \
               any([(key[0]==col or key[1]==col) for colpair in dic_comb for col in colpair])==False:
                dic_comb.append(key)
        print ("Combined twin-column(s): ", dic_comb) # see final twin-column list
        # start 2D Bayesian Gaussian Mixture fit
        for col1, col2 in dic_comb:
            column_data = data[[col1, col2]].values # shape(len-data, 2)
            meta = self._fit_continuous((col1, col2), column_data, dim=2)
        self.output_info += meta['output_info']
        self.output_dimensions += meta['output_dimensions']
        self.meta.append(meta)
        
        # continue with to normal schedule
        for column in data.columns:
            column_data = data[[column]].values
            if column in discrete_columns:
                meta = self._fit_discrete(column, column_data)
            else:
                meta = self._fit_continuous(column, column_data)

            self.output_info += meta['output_info']
            self.output_dimensions += meta['output_dimensions']
            self.meta.append(meta)

    def _transform_continuous(self, column_meta, data, dim=1):
        # these are unchanged info for dim=1 or 2
        components = column_meta['components']
        model = column_meta['model']
        probs = model.predict_proba(data)
        n_opts = components.sum()
        probs = probs[:, components]
        
        opt_sel = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            pp = probs[i] + 1e-6
            pp = pp / pp.sum()
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
        
        # below are values (basecally: features) need to compute twice
        features_ = []
        for idim in range(dim):
            means = model.means_[:, idim].reshape((1, self.n_clusters))
            stds = np.sqrt(model.covariances_[:, idim, idim]).reshape((1, self.n_clusters))
            features = (data[:, idim].reshape((-1, 1)) - means) / (4 * stds)
            features = features[:, components]

            idx = np.arange((len(features)))
            features = features[idx, opt_sel].reshape([-1, 1])
            features = np.clip(features, -.99, .99)
            features_.append(features)
        features_ = np.concatenate(features_, axis=1)
        
        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        print (column_meta['name'], dim, [features_, probs_onehot])
        return [features_, probs_onehot]

    def _transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        return encoder.transform(data)

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self.meta:
            if isinstance(meta['name'], tuple): # is a twin-column
                column_data = data[[meta['name'][0], meta['name'][1]]].values
                values += self._transform_continuous(meta, column_data, dim=2)
                continue
            # back to normal
            column_data = data[[meta['name']]].values
            if 'model' in meta:
                values += self._transform_continuous(meta, column_data)
            else:
                values.append(self._transform_discrete(meta, column_data))

        return np.concatenate(values, axis=1).astype(float)

    def _inverse_transform_continuous(self, meta, data, sigma):
        model = meta['model']
        components = meta['components']

        u = data[:, 0]
        v = data[:, 1:]

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), self.n_clusters)) * -100
        v_t[:, components] = v
        v = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, meta, data):
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data, sigmas):
        start = 0
        output = []
        column_names = []
        for meta in self.meta:
            dimensions = meta['output_dimensions']
            if isinstance(meta['name'], tuple):
                start += dimensions
                continue
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                sigma = sigmas[start] if sigmas else None
                inverted = self._inverse_transform_continuous(meta, columns_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        if self.dataframe:
            output = pd.DataFrame(output, columns=column_names)

        return output

import numpy as np


class MNLogit:
    """ Define the utility and some useful functions """

    def __init__(self, df, utility, availability=None):
        """"
        Initialize the class

        :param df: DataFrame
        """

        self.df = df
        self.utility = utility

        self.params = []

        self.params_choice = {}

        # Sanity check!
        for alt in self.utility.keys():
            for pair in self.utility[alt]:
                if len(pair) == 2:
                    self.params.append(pair[0])
                if pair[0] not in self.params_choice.keys():
                    self.params_choice[pair[0]] = {}

                self.params_choice[pair[0]][alt] = pair[1]

                if isinstance(pair[1], str):
                    if pair[1] not in self.df.columns:
                        raise ValueError('Wrong parameter: {}! Not in the DF!'.format(pair[1]))

        self.params = np.unique(self.params)

        self.params_to_index = {}
        for i, p in enumerate(self.params):
            self.params_to_index[p] = i

        self.choice = {}

        for i, a in enumerate(self.utility.keys()):
            self.choice[i+1] = a

        if availability is None:
            self.availability = {}
            for alt in self.utility.keys():
                self.availability[alt] = np.ones(len(df))
        else:
            self.availability = availability

    def compute_utility(self, x, indices=None, alt=None):
        """
        Compute the utility for a given alternative.

        If None are given, compute the utility for all alternatives

        :param x: Value for the parameters
        :param indices: Indices for which we compute the utility
        :param alt: String with an alternative
        :return: Either a float or a dict
        """

        if indices is None:
            indices = np.array(range(len(self.df)))

        if not isinstance(indices, list)  and not isinstance(indices, np.ndarray):
            indices = [indices]

        if alt is None:
            utility = {}
            for alt in self.utility.keys():
                ut = np.zeros(len(indices))
                for pair in self.utility[alt]:
                    if len(pair) > 2:
                        beta = pair[2]
                    else:
                        idx = self.params_to_index[pair[0]]
                        beta = x[idx]

                    if isinstance(pair[1], str):
                        ut += np.array(beta*np.array(self.df[pair[1]])[indices])
                    else:
                        ut += beta*np.ones(len(indices))*pair[1]

                utility[alt] = ut
        else:
            if alt not in self.utility.keys():
                raise ValueError('Wrong alternative!')

            utility = np.empty(len(indices))
            for pair in self.utility[alt]:
                if len(pair) > 2:
                    beta = pair[2]
                else:
                    idx = self.params_to_index[pair[0]]
                    beta = x[idx]

                if isinstance(pair[1], str):
                    utility += np.array(beta * np.array(self.df[pair[1]]))[indices]
                else:
                    utility += beta * np.ones(len(indices)) * pair[1]

        return utility

    def probabilities(self, x, indices=None):
        """
        Compute probabilities for given parameters

        :param x: array with values of parameters
        :param indices: Array with indices
        :return:
        """

        proba = {}

        utilities = self.compute_utility(x, indices)

        if indices is None:
            indices = np.array(range(len(self.df)))

        tmp = np.array([self.availability[u][indices]*np.exp(utilities[u]) for u in utilities])

        denom = np.sum(tmp, axis=0)

        for a in self.utility.keys():
            proba[a] = np.exp(utilities[a]) / (denom+1e-8)

        return proba

    def loglikelihood(self, x, indices=None):
        """
        Log Likelihood for given parameters

        :param x: parameters
        :return:
        """

        if indices is None:
            indices = np.array(range(len(self.df)))

        if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
            indices = [indices]

        proba = self.probabilities(x, indices)

        log_vals = [np.log(proba[self.choice[c]][i]) for i, c in enumerate(np.array(self.df['CHOICE'])[indices])]

        return 1/len(indices)*np.sum(log_vals)

    def negloglikelihood(self, x, indices=None):

        return -self.loglikelihood(x, indices)

    def reg_loglikelihood(self, x, indices=None):
        """
        Log likelihood with Regularization term

        :param x:
        :param indices:
        :return:
        """

        res = self.loglikelihood(x, indices)

        res = res - np.sum(x**2)

        return res

    def reg_negloglikelihood(self, x, indices=None):

        return  -self.reg_loglikelihood(x, indices)

    def num_grad(self, x, indices=None):
        """
        Compute the gradient with finite differences

        :param x: parameters
        :param indices: indices
        :return:
        """

        eps = 1e-6

        f = lambda param: self.loglikelihood(param, indices)

        # Size the problem, i.e. nbr of parameters
        n = len(x)

        # Prepare the vector for the gradient
        grad = np.zeros(n)

        # Prepare the array to add epsilon to.
        dx = np.zeros(n)

        # Go through all parameters
        for i in range(len(x)):
            # Add epsilon to variate a parameter
            dx[i] += eps

            # Central finite differences
            grad[i] = -(f(x + dx) - f(x - dx)) / (2 * eps)

            # Set back to 0
            dx[i] = 0

        return grad

    def num_hessian(self, x, indices=None):
        """
        Compute the hessian with finite differences

        :param x: parameters
        :param indices: indices
        :return:
        """

        eps = 1e-6

        grad = lambda param: self.num_grad(param, indices)

        # Size the problem, i.e. nbr of parameters
        n = len(x)

        # Prepare the vector for the gradient
        hess = np.zeros((n,n))

        # Prepare the array to add epsilon to.
        dx = np.zeros(n)

        # Go through all parameters
        for i in range(n):
            # Add epsilon to variate a parameter
            dx[i] += eps

            # Compute the gradient with forward and backward difference
            grad_plus = grad(x+dx)
            grad_minus = grad(x-dx)

            # Central finite difference
            hess[i,:] = -(grad_plus - grad_minus)/(2*eps)

            # Set back to 0
            dx[i] = 0

        return hess

    def dVdBeta(self, param, indices, mat=False):
        param_choice = self.params_choice[param]

        dVdBeta = {}
        dVdBeta_mat = []

        for id_ in self.choice:
            str_ = self.choice[id_]

            try:
                val = param_choice[str_]

                if isinstance(val, str):
                    vals = np.array(self.df[val].iloc[indices])
                else:
                    vals = np.ones(len(indices))

            except:
                vals = np.zeros(len(indices))

            dVdBeta[str_] = vals
            dVdBeta_mat.append(vals)

        if mat:
            return dVdBeta, np.array(dVdBeta_mat)
        else:
            return dVdBeta

    def first_derivative(self, beta_k, x, indices=None):

        dVdBeta_k = self.dVdBeta(beta_k, indices)

        proba = self.probabilities(x, indices)

        val = 0

        for id_ in self.choice.keys():
            str_ = self.choice[id_]

            y = np.array((self.df['CHOICE'].iloc[indices] == id_)).astype(dtype=int)
            p = proba[str_]

            val += np.sum((y - p) * dVdBeta_k[str_])

        return val / len(indices)

    def grad(self, x, indices=None):

        if indices is None:
            indices = np.array(range(len(self.df)))

        return np.array([self.first_derivative(p, x, indices) for p in self.params])

    def neg_grad(self, x, indices=None):

        return -self.grad(x, indices)

    def second_derivative(self, beta_k, beta_l, x, indices=None):

        if indices is None:
            indices = np.array(range(len(self.df)))
    
        dVdBeta_k, dVdBeta_k_mat = self.dVdBeta(beta_k, indices, True)
        dVdBeta_l, dVdBeta_l_mat = self.dVdBeta(beta_l, indices, True)
    
        proba = self.probabilities(x, indices)
    
        proba_mat = np.array([proba[ch] for ch in proba.keys()])
    
        sum_beta_k = np.sum(dVdBeta_k_mat * proba_mat, axis=0)
        sum_beta_l = np.sum(dVdBeta_l_mat * proba_mat, axis=0)
    
        val = 0
    
        for id_ in self.choice.keys():
            k = dVdBeta_k[self.choice[id_]] - sum_beta_k
            l = dVdBeta_l[self.choice[id_]] - sum_beta_l
    
            val += np.sum(proba[self.choice[id_]] * k * l)
    
        return -val / len(indices)
    
    def hessian(self, x, indices=None):

        if indices is None:
            indices = np.array(range(len(self.df)))
    
        hess = np.array([self.second_derivative(beta_k, beta_l, x, indices) for beta_k in self.params for beta_l in self.params])
    
        return hess.reshape(len(self.params), len(self.params))

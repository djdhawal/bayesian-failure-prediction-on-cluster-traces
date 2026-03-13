class GaussianHMMDecoder:
    """
    Gaussian Hidden Markov Model decoder using parameters
    estimated from a Bayesian HMM (NumPyro).

    Implements:
    - emission likelihood
    - forward algorithm
    - viterbi decoding
    - dataframe utilities
    """

    def __init__(self, pi, A, mu, sigma):
        """
        Parameters
        ----------
        pi : (K,)
            Initial state probabilities

        A : (K,K)
            Transition matrix

        mu : (K,D)
            Mean of Gaussian emissions

        sigma : (K,D)
            Standard deviation (diagonal covariance)
        """

        self.pi = np.asarray(pi)
        self.A = np.asarray(A)
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)

        self.K = self.pi.shape[0]
        self.D = self.mu.shape[1]

        # log-space versions (used internally)
        self.logpi = np.log(self.pi + 1e-12)
        self.logA = np.log(self.A + 1e-12)
    
    # Emission Probability
    def log_normal_diag(x, mu, sigma):
        """
        x: observation sequence (T,D)
        mu/sigma: Gaussian parameters for each time and state (K,D)
        returns log p(x_t | z=k) as (T,K)
        """
        # Expand observations so they broadcast across
        x = x[:, None, :]               # (T,1,D)
        mu = mu[None, :, :]             # (1,K,D)
        sg = sigma[None, :, :]          # (1,K,D)

        # Compute Gaussian log-density and sum across feature dimension D
        return (-0.5 * (((x - mu) / sg) ** 2 + 2*np.log(sg) + np.log(2*np.pi))).sum(-1)

    # Forward Algorithm
    def forward_probs(X, pi, A, mu, sigma):
        """
        X: observation sequence (T, D)
        pi: initial state probabilities (K,)
        A: transition matrix (K, K)
        mu, sigma: Gaussian parameters for each state (K, D)
        Returns gamma_filt: filtered state probabilities (T, K)
        """
    
        T = X.shape[0]
        K = pi.shape[0]
    
        # Compute log emission probabilities log p(x_t | z_t=k)
        logB = log_normal_diag(X, mu, sigma)   # (T, K)
    
        # Convert initial and transition probabilities to log-space
        logpi = np.log(pi + 1e-12)             # (K,)
        logA = np.log(A + 1e-12)               # (K, K)
    
        # Allocate forward probability matrix
        log_alpha = np.zeros((T, K), dtype=np.float64)
    
        # t = 0
        log_alpha[0] = logpi + logB[0]
        log_alpha[0] -= logsumexp(log_alpha[0])   # normalize
    
        # t = 1, ..., T-1
        for t in range(1, T):
            # for each current state k:
            # log_alpha[t, k] = logB[t, k] + logsumexp_j(log_alpha[t-1, j] + logA[j, k])
            trans_scores = log_alpha[t - 1][:, None] + logA   # (K, K)
            log_alpha[t] = logB[t] + logsumexp(trans_scores, axis=0)
            log_alpha[t] -= logsumexp(log_alpha[t])           # normalize
    
        gamma_filt = np.exp(log_alpha)   # (T, K), rows sum to ~1
        return gamma_filt
    

    # Viterbi Algorithm
    def viterbi_path(X, pi, A, mu, sigma):
        """
        X: observation sequence (T, D)
        pi: initial state probabilities (K,)
        A: transition matrix (K, K)
        mu, sigma: Gaussian emission parameters (K, D)
        Returns path: most likely hidden state sequence (T,)
        """
    
        T = X.shape[0]
        K = pi.shape[0]
    
        # Compute emission log probabilities
        logB = log_normal_diag(X, mu, sigma)   # (T, K)
        logpi = np.log(pi + 1e-12)
        logA = np.log(A + 1e-12)
    
        delta = np.zeros((T, K), dtype=np.float64)
        psi = np.zeros((T, K), dtype=np.int32)
    
        # initialization
        delta[0] = logpi + logB[0]
    
        # recursion
        for t in range(1, T):
            for k in range(K):
                scores = delta[t - 1] + logA[:, k]
                psi[t, k] = np.argmax(scores)
                delta[t, k] = np.max(scores) + logB[t, k]
    
        # backtrack
        path = np.zeros(T, dtype=np.int32)
        path[-1] = np.argmax(delta[-1])
    
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

        
    # Sequence Utilities
    def decode_forward(self, X):
        """Wrapper for forward probabilities"""
        return self.forward_probs(X)

    def decode_viterbi(self, X):
        """Wrapper for viterbi decoding"""
        return self.viterbi_path(X)

    
    # DataFrame Helpers
     def add_forward_state_probs(df, feature_set, pi, A, mu, sigma, time_col="start_time"):
        """
        df: dataframe containing job observations
        feature_set: columns used as HMM features
        pi, A, mu, sigma: learned HMM parameters
        Returns dataframe with state probability columns added
        """
         
        df_out = df.copy()
    
        K = len(pi)
        # Create column names for each hidden state probability
        prob_cols = [f"hmm_state_{k}_prob" for k in range(K)]
    
        # initialize probability columns
        for col in prob_cols:
            df_out[col] = np.nan
    
        # preserve original row identity so we can write back in correct order
        df_out["_orig_idx"] = np.arange(len(df_out))
    
        for (cid, iid), g in df_out.groupby(["collection_id", "instance_index"], sort=False):
            # Process each job instance in time order
            g_sorted = g.sort_values(time_col)
    
            # Extract feature sequence
            X = g_sorted[feature_set].to_numpy(dtype=np.float32)
    
            if X.shape[0] == 0:
                continue
    
            # Compute filtered state probabilities
            probs = forward_probs(X, pi, A, mu, sigma)   # (T, K)
    
            # assign back row-by-row in sorted time order
            df_out.loc[g_sorted.index, prob_cols] = probs
    
        df_out = df_out.drop(columns="_orig_idx")
        return df_out
         

    def add_viterbi_state_column(df, feature_set, pi, A, mu, sigma, time_col="start_time"):
        """
        df: dataframe containing job observations
        feature_set: columns used as HMM features
        pi, A, mu, sigma: learned HMM parameters
        Returns dataframe with a new column "viterbi_state"
        """
    
        df_out = df.copy()
        df_out["viterbi_state"] = np.nan
    
        for (cid, iid), g in df_out.groupby(["collection_id", "instance_index"], sort=False):
            g_sorted = g.sort_values(time_col)
    
            X = g_sorted[feature_set].to_numpy(dtype=np.float32)
    
            if len(X) == 0:
                continue
    
            path = viterbi_path(X, pi, A, mu, sigma)
    
            df_out.loc[g_sorted.index, "viterbi_state"] = path
    
        df_out["viterbi_state"] = df_out["viterbi_state"].astype("Int64")
        return df_out

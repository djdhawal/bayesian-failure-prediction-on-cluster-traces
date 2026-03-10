import pandas as pd
from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.handlers import mask
from numpyro.infer import HMC
from numpyro.ops.indexing import Vindex
import numpy as np
import funsor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from numpyro import handlers
from numpyro.infer import SVI, TraceEnum_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import optax_to_numpyro
from numpyro import util
import optax


class ModelMania():
    def __init__(self):
            self.sequences = []
            self.lengths = []
            self.model = None
            self.results = None
            self.guide = None
            self.rng_key = random.PRNGKey(0)
            self.steps = 15000
            self.args = { 'h_dim' : 3 }
            pass

    def get_task(self, df, time_col='start_time', target_pair = None):
        #_df_task = get_task(df_train, target_pair=(399280623256, 758))
        (collection_id, instance_index) = target_pair
        if collection_id is not None:
            df = df[df['collection_id'] == collection_id]
        if instance_index is not None:
            df = df[df['instance_index'] == instance_index]
        
        df = df.sort_values(by=time_col, ascending=True)
        return df

    def create_sequences(self, df,
                        feature_columns_list=None,
                        pairs = None,
                        window_size=100):
    
        assert pairs is not None, "pairs must be provided"

        sequences =[]
        lengths =[]
        for target_pair in pairs:
            #print(*(target_pair))
            task_data = self.get_task(df, target_pair=target_pair)  # should return numpy array of shape (T, 4)

            features = task_data[feature_columns_list].values.astype(np.float32)

            n_windows = len(features) // window_size
            for w in range(n_windows):
                window = features[w * window_size : (w + 1) * window_size]
                sequences.append(window)
                lengths.append(window_size)

        # Convert to JAX arrays
        self.sequences = jnp.array(np.stack(sequences))  # (num_total_windows, 100, 4)

        return self.sequences, self.lengths


    def plot_priors(self, feature_set, df):
        '''
            plots histogram of feature set against emmision priors
        '''
        cols = list({c for c in feature_set if c in df.columns and not df[c].isna().all()})
        if not cols: print("no numerical columns found to plot")
        else:
            n_cols, n_rows = 3, (len(cols)+2)//3
            plt.figure(figsize=(n_cols*6, n_rows*5))
            for i,c in enumerate(cols,1):
                plt.subplot(n_rows,n_cols,i); d=df[c].dropna(); normed=c.endswith('_normed')
                sns.histplot(d, kde=True, stat="density" if normed else None, bins=50)
                if normed:
                    x=np.linspace(*plt.xlim(),100); plt.plot(x,norm.pdf(x,0,1),'r',lw=2,label='Prior N(0,1)'); plt.legend()
                plt.title(f'Distribution of {c}' + (' (Normalized)' if normed else ''),fontsize=12)
                plt.xlabel(c,fontsize=10); plt.ylabel('Density' if normed else 'Frequency',fontsize=10)
            plt.tight_layout(); plt.show()

    def model_for_guide(self, *args, **kwargs):
        '''
            hiding the hidden state from inference
        '''
        with handlers.block(hide=["z"]):
            self.model_gaussian_hmm(*args, **kwargs)

    def initialize_inference(self):
        '''
            returns initialized model ready to train.
        '''
        # autonormal guide foe svi
        self.guide = AutoNormal(self.model_for_guide)

        # creating custom optimizer from optax, decaying clipped adam at 1k steps
        scheduler = optax_to_numpyro(
            optax.chain(
                optax.clip_by_global_norm(5.0),
                optax.adam(learning_rate=optax.exponential_decay(
                    init_value=0.005,
                    transition_steps=1000,
                    decay_rate=0.5
                ))
            )
        )
        
        self.model = SVI(self.model_gaussian_hmm, self.guide, scheduler, TraceEnum_ELBO())

        return self.model
    

    def run_model_inference(self):
        
        self.results = self.model.run(
            self.rng_key,
            self.steps,               # num_steps
            self.sequences,           # positional arg: sequences
            self.lengths,             # positional arg: lengths
            self.args,                # positional arg: args
        )

        return self.results

    
    def model_gaussian_hmm(self, sequences, lengths, args, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        K = args['h_dim']

        with mask(mask=include_prior):

            pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(K)))

            probs_z = numpyro.sample(
                "probs_z",
                dist.Dirichlet(jnp.ones(K)).expand([K]).to_event(1)
            )
            mu = numpyro.sample(
                "mu",
                dist.Normal(0.0, 1.0)
                .expand([K, data_dim])
                .to_event(2)
            )

            sigma = numpyro.sample(
                "sigma",
                dist.HalfNormal(1.0)
                .expand([K, data_dim])
                .to_event(2)
            )

        def transition_fn(carry, x_t):
            z_prev, t = carry
            with numpyro.plate("sequences", num_sequences, dim=-1):
                with mask(mask=(t < lengths)):
                    z = numpyro.sample(
                            "z",
                            dist.Categorical(jnp.where(t == 0, pi, probs_z[z_prev])),
                            infer={"enumerate": "parallel"},
                    )
                    numpyro.sample(
                        "obs",
                        dist.Independent(
                            dist.Normal(
                                Vindex(mu)[z, :],
                                Vindex(sigma)[z, :] + 1e-6
                            ), 1
                        ),
                        obs=x_t,
                    )
            return (z, t + 1), None

        z_init = jnp.zeros(num_sequences, dtype=jnp.int32)
        scan(transition_fn, (z_init, 0), jnp.swapaxes(sequences, 0, 1))  

    def validate_model_structure(self):
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                self.model_gaussian_hmm(self.sequences, self.lengths, self.args)
            print(util.format_shapes(tr))
        
    def render_model(self):
        model = numpyro.render_model(self.model_gaussian_hmm,
                    model_args=(self.sequences,self.lengths,self.args),
                    render_distributions=True)
        return model
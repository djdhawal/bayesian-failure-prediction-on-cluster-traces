import pandas as pd
import numpy as np
import regex as re

class FeatureEngineer():
    def __init__(self):
        self._NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    def fit_log_norm(df, cols, eps=1e-7):
        # fitting normalization parameters (TRAIN ONLY)
        params = {}
        for col in cols:
            x = df[col].astype(float)
            x_logged = np.log(x + eps)
            mu = x_logged.mean()
            sigma = x_logged.std()
            if sigma < 1e-8:
                sigma = 1e-8
    
            params[col] = {
                "eps": eps,
                "mean": mu,
                "std": sigma
            }
        return params

        
    def apply_log_norm(df, cols, params):
        # apply normalization TRAIN + TEST
        for col in cols:
            eps = params[col]["eps"]
            mu = params[col]["mean"]
            sigma = params[col]["std"]
    
            x = df[col].astype(float)
            df[f"{col}_logged"] = np.log(x + eps)
            df[f"{col}_logged_normed"] = (df[f"{col}_logged"] - mu) / sigma
        return df

    import numpy as np, regex as re

    def _fast_parse_col(self, series, expected_len):
        """vectorised parse, returns float64 array"""
        
        out = np.full((len(series), expected_len), np.nan, dtype=np.float64)
        vals = series.values

        for i in range(len(vals)):
            v = vals[i]
            try:
                if isinstance(v, str):
                    nums = self._NUM.findall(v)
                    k = min(len(nums), expected_len)
                    for j in range(k):
                        out[i, j] = float(nums[j])
                elif isinstance(v, (list, np.ndarray)):
                    k = min(len(v), expected_len)
                    for j in range(k):
                        out[i, j] = float(v[j])
            except Exception:
                pass
        return out

    def clean_cpu_usage_distribution(self, df):
        a = self._fast_parse_col(df["cpu_usage_distribution"],11)
        df[[f"cpu_p{i}" for i in range(0,101,10)]] = a
        df["cpu_burstiness"] = a[:,9]-a[:,1]
        return df

    def extract_tail_cpu_features(self, df):
        '''
            tail cpu is an array and sometimes string, we extract the usage telemetry 
            and create features from it
        '''
        a = self._fast_parse_col(df["tail_cpu_usage_distribution"],9)
        with np.errstate(all="ignore"):
            c = (~np.isnan(a)).sum(1)
            df["tail_cpu_mean"] = np.where(c, np.nanmean(a,1), np.nan)
            df["tail_cpu_max"]  = np.where(c, np.nanmax(a,1), np.nan)
            df["tail_cpu_p90"]  = np.where(c, np.nanpercentile(a,90,1), np.nan)
            df["tail_cpu_nonzero_frac"] = np.where(c, np.nansum(a>0,1)/c, np.nan)
        return df

    def remove_pairs_with_nulls(self, df_train, cols = None):
        '''
            does what it says
        '''
        #df_train.isnull().sum()
        #cols = ["cpu_p90_logged_normed","cpu_burstiness_logged_normed"]
        bad_pairs = df_train.loc[
            df_train[cols].isna().any(axis=1),
            ["collection_id","instance_index"]
        ].drop_duplicates() #.shape[0]

        df_train = df_train.merge(bad_pairs, on=["collection_id","instance_index"], how="left", indicator=True)
        df_train = df_train[df_train["_merge"] == "left_only"].drop(columns="_merge")
        return df_train    

    def clean_log_transforms(
            self, 
            df: pd.DataFrame, 
            epsilon: float = 1e-7, 
            cols_to_tansform: list = None) -> pd.DataFrame:
        """
            applies log transform to list of columns passed as cols_to_transform
            inserts log columns immediately to the right.
        """
        out = df.copy()

        for col in cols_to_tansform:
            if col not in out.columns:
                continue

            out[col] = pd.to_numeric(out[col], errors="coerce")

            log_col = f"log_{col}"
            log_values = np.log(out[col] + epsilon)

            insert_at = out.columns.get_loc(col) + 1
            out.insert(insert_at, log_col, log_values)

        return out
    


import pandas as pd
import numpy as np
import regex as re

class FeatureTransformer():
    def clean_cpu_usage_distribution(
        self,    
        df: pd.DataFrame,
        col: str = "cpu_usage_distribution",
        expected_len: int = 11,
        keep_all_percentiles: bool = True,
        drop_original: bool = True ) -> pd.DataFrame:
        """
        Parses cpu_usage_distribution which is a list of floats
        Produces cpu_p0, cpu_p10, ..., cpu_p100
        and cpu_burstiness = cpu_p90 - cpu_p10
        """

        out = df.copy()
        if col not in out.columns:
            return out

        def parse_dist(val, expected_len=expected_len):
            try:
                if isinstance(val, (list, np.ndarray, tuple)):
                    vals = list(val)
                elif isinstance(val, str):
                    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
                    vals = [float(x) for x in nums]
                else:
                    return [np.nan] * expected_len

                # pad/truncate to expected_len
                vals = vals[:expected_len] + [np.nan] * max(0, expected_len - len(vals))
                return vals
            except Exception:
                return [np.nan] * expected_len

        dist_parsed = out[col].apply(parse_dist)

        pct_cols = [f"cpu_p{i}" for i in range(0, 101, 10)]  # p0..p100 
        dist_df = pd.DataFrame(dist_parsed.tolist(), columns=pct_cols, index=out.index)

        if keep_all_percentiles:
            # insert all percentile columns right after the original col
            insert_at = out.columns.get_loc(col) + 1
            for j, c in enumerate(pct_cols):
                out.insert(insert_at + j, c, dist_df[c])
        else:
            # only keep the ones you actually use
            out["cpu_p10"] = dist_df["cpu_p10"]
            out["cpu_p50"] = dist_df["cpu_p50"]
            out["cpu_p90"] = dist_df["cpu_p90"]
    
        out["cpu_burstiness"] = out["cpu_p90"] - out["cpu_p10"]

        # not dropping the op column for now, we'll manually select features
        # if drop_original:
            # out = out.drop(columns=[col])

        return out
    

    def clean_start_time_to_datetime(
            self,
            df: pd.DataFrame, 
            cols: list = None) -> pd.DataFrame:
        """
        Converts Unix timestamp columns to pandas datetime.
        Inserts the new datetime columns immediately to the right of the original column.
        """
        out = df.copy()

        unit = 'us'

        for col in cols:
            if col not in out.columns:
                continue
        
            new_col_name = col + "_datetime"
            datetime_values = pd.to_datetime(out[col], unit=unit, errors="coerce")

            # find index of original column
            col_index = out.columns.get_loc(col)

            # insert new column immediately after it
            out.insert(col_index + 1, new_col_name, datetime_values)

        return out
    

    def clean_log_transforms(
            self, 
            df: pd.DataFrame, 
            epsilon: float = 1e-7, 
            cols_to_tansform: list = None) -> pd.DataFrame:
        """
        Applies log transform to list of columns passed as cols_to_transform
        Inserts log columns immediately to the right.
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
    


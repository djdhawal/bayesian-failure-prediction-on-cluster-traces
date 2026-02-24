import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb
import os

class data_loader():

    def __init__(self, base_url='gs://clusterdata_2019_a', threads=None, max_workers=None):
        self.base_url = base_url
        # max_workers × threads ≈ 2x number of CPU cores; assuming 10 cores
        num_cpus = os.cpu_count()  # logical cores
        self.threads = threads or max(1, num_cpus // 4)
        self.max_workers = max_workers or max(1, num_cpus // self.threads)


    def parallel_load_test(self, fn, shards=0, max_workers=16, batch_size=10, **kwargs) -> pd.DataFrame:
        '''
        function to run another function that returns a dataframe across concurrent futures.
            args : 
                fn : function which will be executed by concurrent futures
                shards : number of shards we want to read from the gcs bucket
                max-workers : number of parallel workers
                batch_size : read shards in batches of batch_size\
        '''
        batches = list(range(0, shards, batch_size))  # [0, 10, 20, ...]
        results = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(fn, start, min(start + batch_size - 1, shards - 1), **kwargs): i
                for i, start in enumerate(batches)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                print(f"batch {i+1}/{len(batches)}")
        return pd.concat(results, ignore_index=True)


    def dck_db_load(self, shard_range_start=None, shard_range_end=None, cols=None, collection_ids=None, instance_indexes=None) -> pd.DataFrame:
        assert shard_range_start is not None, "shard_range_start cannot be None"
        assert shard_range_end is not None, "shard_range_end cannot be None"
        assert cols is not None, "cols cannot be None"
        assert collection_ids is not None, "collection_ids cannot be None"
        assert instance_indexes is not None, "instance_indexes cannot be None"


        con = duckdb.connect()
        con.sql("INSTALL httpfs; LOAD httpfs;")
        con.sql(f"SET threads = {self.threads};")

        dfs=[]
        print(f"processing {shard_range_start} to {shard_range_end}...")
        for char in [str(i).zfill(5) for i in range(shard_range_start,shard_range_end+1)]:
            print(f"processing {char}...")
            df = con.sql(f"""
                SELECT {cols}

                FROM read_parquet('{self.base_url}/instance_usage-0000000{char}.parquet.gz')
                WHERE collection_id IN ({collection_ids})
                AND instance_index IN ({instance_indexes})
                ORDER BY collection_id, instance_index, start_time
                --LIMIT 10000
            """).df()
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
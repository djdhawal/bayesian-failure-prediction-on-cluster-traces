
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb
import os

class DataLoader():

    def __init__(self, base_url='gs://clusterdata_2019_a', threads=None, max_workers=None):
        self.base_url = base_url
        # max_workers × threads ≈ 2x number of CPU cores; assuming 10 cores
        num_cpus = os.cpu_count()  # logical cores
        self.threads = threads or max(1, num_cpus // 4)
        self.max_workers = max_workers or max(1, num_cpus // self.threads)


    def parallel_load(self, fn, shards=0, max_workers=None, batch_size=10, **kwargs) -> pd.DataFrame:
        '''
        function to run another function (that returns a dataframe) across concurrent futures.
            args :
                fn : function which will be executed by concurrent futures
                shards : number of shards we want to read from the gcs bucket
                max-workers : number of parallel workers
                batch_size : read shards in batches of batch_size\
        '''
        max_workers = max_workers or self.max_workers
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


    def dck_db_load(
                      self,
                      shard_range_start=None,
                      shard_range_end=None,
                      query=None,
                      cols=None,
                      target_collection_ids=None,
                      target_instance_indexes=None,
                      target_pairs=None
                      ) -> pd.DataFrame:

        '''
        function to initialize a duckdb connection and query shards between 'shard_range_start' and 'shard_range_end' from gcs bucket.
        args:
            shard_range_start : start reading shards from here
            shard_range_end : stop reading here
            query : string containing the query to run
            cols : string containing comma separated cols to pull (goes inside the query)
            target_collection_ids : string containing comma separated collection ids to filter on (goes inside the query)
            target_instance_indexes : string containing comma separated instance idxs to filter on (goes inside the query) // this might probably be removed
            target_pairs : string containing comma separated pairs of (collection_id, instance_index) to filter on (goes inside the query)
        '''
        assert shard_range_start is not None, "shard_range_start cannot be None"
        assert shard_range_end is not None, "shard_range_end cannot be None"
        assert query is not None, "query cannot be None"
        assert cols is not None, "cols cannot be None"
        if (query == 'instance_usage') | (query == 'collection_events'):
          assert target_collection_ids is not None, "collection_ids cannot be None"
        if query == 'instance_usage':
          assert target_instance_indexes is not None, "instance_indexes cannot be None"
        if query == 'instance_usage_1':
          assert target_pairs is not None, "instance_pairs cannot be None"

        con = duckdb.connect()
        con.sql("INSTALL httpfs; LOAD httpfs;")
        con.sql(f"SET threads = {self.threads};")

        try:
            dfs=[]
            print(f"processing {shard_range_start} to {shard_range_end}...")
            for i in range(shard_range_start,shard_range_end+1):
                char = str(i).zfill(5)
                print(f"processing {char}...")

                query_set = {
                    'collection_events' : f"""
                            SELECT {cols}
                            FROM read_parquet('{self.base_url}/collection_events-0000000{char}.parquet.gz')
                            WHERE collection_id IN ({target_collection_ids})
                            LIMIT 10000
                        """,
                    'instance_events' : f"""
                            SELECT {cols}
                            FROM read_parquet('{self.base_url}/instance_events-0000000{char}.parquet.gz')
                            WHERE collection_id IN ({target_collection_ids})
                            --LIMIT 10000
                        """,
                    'instance_usage' : f"""
                            SELECT {cols}

                            FROM read_parquet('{self.base_url}/instance_usage-0000000{char}.parquet.gz')
                            WHERE collection_id IN ({target_collection_ids})
                            AND instance_index IN ({target_instance_indexes})
                            ORDER BY collection_id, instance_index, start_time
                            --LIMIT 10000
                        """,
                    'instance_usage_1' : f"""
                            SELECT {cols}
                            FROM read_parquet('{self.base_url}/instance_usage-0000000{char}.parquet.gz')
                            WHERE (collection_id, instance_index) IN ({target_pairs})
                            ORDER BY collection_id, instance_index, start_time
                            --LIMIT 10000"""
                }

                df = con.sql(query_set[query]).df()
                dfs.append(df)
        finally:
            con.close()

        return pd.concat(dfs, ignore_index=True)
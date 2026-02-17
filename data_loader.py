
import pyarrow.parquet as pq
import pandas as pd

# sweep instance_usage in batches, collect all rows
usage_file = pq.ParquetFile("instance_usage-000000000000.parquet.gz")

usage_frames = []
for batch in usage_file.iter_batches(batch_size=100_000):
    usage_frames.append(batch.to_pandas())
    total = sum(len(f) for f in usage_frames)
    # Safety cap — adjust as needed
    if total > 500_000:
        print(f"Stopping early with {total} usage rows")
        break

usage_sample = pd.concat(usage_frames, ignore_index=True)
## print(f"usage_sample: {usage_sample.shape}")
## print(f"Memory: {usage_sample.memory_usage(deep=True).sum() / 1e6:.1f} MB")
usage_sample.head()


# Step 2: Extract unique (collection_id, instance_index) pairs
pair_df = usage_sample[['collection_id', 'instance_index']].drop_duplicates()
sampled_ids = set(pair_df['collection_id'].unique())
## print(f"Unique (collection_id, instance_index) pairs: {len(pair_df)}")
## print(f"Unique collection_ids: {len(sampled_ids)}")

# Step 3: Filter instance_events by matching (collection_id, instance_index) pairs
ie_file = pq.ParquetFile("instance_events-000000000000.parquet.gz")

ie_frames = []
for batch in ie_file.iter_batches(batch_size=100_000):
    chunk = batch.to_pandas()
    filtered = chunk.merge(pair_df, on=['collection_id', 'instance_index'], how='inner')
    if len(filtered):
        ie_frames.append(filtered)
    # Safety cap
    total = sum(len(f) for f in ie_frames)
    if total > 50_000:
        print(f"Stopping early with {total} instance_event rows")
        break

ie_sample = pd.concat(ie_frames, ignore_index=True) if ie_frames else pd.DataFrame()
## print(f"instance_events matched: {ie_sample.shape}")
ie_sample.head()



# Step 4: Filter collection_events by matching collection_ids
ce_file = pq.ParquetFile("collection_events-000000000000.parquet.gz")

ce_frames = []
for batch in ce_file.iter_batches(batch_size=100_000):
    chunk = batch.to_pandas()
    filtered = chunk[chunk['collection_id'].isin(sampled_ids)]
    if len(filtered):
        ce_frames.append(filtered)

    total = sum(len(f) for f in ce_frames)
    if total > 50_000:
        print(f"Stopping early with {total} collection_event rows")
        break

ce_sample = pd.concat(ce_frames, ignore_index=True) if ce_frames else pd.DataFrame()
## print(f"collection_events matched: {ce_sample.shape}")
ce_sample.head()


# 3-way join: instance_usage ⟶ instance_events ⟶ collection_events
# Join 1: usage ↔ instance_events on (collection_id, instance_index)
merged = usage_sample.merge(ie_sample, on=['collection_id', 'instance_index'], how='inner', suffixes=('', '_ie'))
print(f"usage ⨝ instance_events: {merged.shape}")

# Join 2: bring in collection_events on collection_id
full = merged.merge(ce_sample, on='collection_id', how='inner', suffixes=('', '_ce'))
## print(f"Full 3-way join: {full.shape}")

# Quick memory check
mem_mb = full.memory_usage(deep=True).sum() / 1e6
## print(f"Memory usage: {mem_mb:.1f} MB")
full.head()
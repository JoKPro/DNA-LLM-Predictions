import os
import pickle
import pandas as pd
import polars as pl


def get_species_table(sequence_dir, species_dir, file_name="embeddings_downstream.csv", to_polars=False): 
    """Get the embeddings table for a specific species.
    """
    seq_path = os.path.join(sequence_dir, species_dir, file_name)
    with open(seq_path, "rb") as f: 
        seq = pickle.load(f)
        
    if to_polars: 
        seq = pl.from_pandas(seq)
    return seq
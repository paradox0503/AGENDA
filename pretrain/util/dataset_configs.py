# coding = utf-8
from .conf import Configuration
import os
config = Configuration(os.path.join(os.path.dirname(__file__), '../conf/example.json'))
conf_size_train = config.getHP('size_train')
conf_size_val = config.getHP('size_val')
conf_size_db = config.getHP('size_db')
class DatasetConfig:
    def __init__(self, name, path_db, dim_seq, size_train, size_val, size_db, index_name):
        self.name = name
        self.path_db = path_db
        self.dim_seq = dim_seq
        self.size_train = size_train
        self.size_val = size_val
        self.size_db = size_db
        self.index_name = index_name
class EmbedConfig:
    def __init__(self, name, dataset_path, query_path, dim_seq, size_query,query_index_name=0):
        self.name = name
        self.dataset_path = dataset_path
        self.query_path = query_path
        self.dim_seq = dim_seq
        self.size_query = size_query
        self.query_index_name = query_index_name
DATASET_CONFIGS = [
    DatasetConfig("Astro", "/data/astro-dataset.bin", 256, size_train=conf_size_train, size_val=conf_size_val, size_db=conf_size_db, index_name=0),
    DatasetConfig("Deep1B", "/data/deep1b-dataset.bin", 96, size_train=conf_size_train, size_val=conf_size_val, size_db=conf_size_db, index_name=1),
    DatasetConfig("F5", "/data/F5-dataset.bin", 256, size_train=conf_size_train, size_val=conf_size_val, size_db=conf_size_db, index_name=2),
    # DatasetConfig("F10", "/data/F10-dataset.bin", 256, 2000, 1000, 10000, index_name=3),
    DatasetConfig("Randwalk", "/data/origin-dataset.bin", 256, size_train=conf_size_train, size_val=conf_size_val, size_db=conf_size_db, index_name=4),
    DatasetConfig("Sald", "/data/sald-dataset.bin", 128, size_train=conf_size_train, size_val=conf_size_val, size_db=conf_size_db, index_name=5),
]# DATASET_CONFIGS
embed_CONFIGS = [
    EmbedConfig("Astro", "data_big/astro-dataset.bin", "data_big/astro-query.bin", 256, 100,query_index_name=0),
    EmbedConfig("Deep1B", "data_big/deep1b-dataset.bin", "data_big/deep1b-query.bin", 96, 1000,query_index_name=1),
    EmbedConfig("F5", "data_big/F5-dataset.bin", "data_big/F5-query.bin", 256, 1000,query_index_name=2),
    # EmbedConfig("F10", "data_big/F10-dataset.bin", "data_big/F10-query.bin", 256, 1000,query_index_name=3),
    EmbedConfig("Randwalk", "data_big/origin-dataset.bin", "data_big/origin-query.bin", 256, 1000,query_index_name=4),
    EmbedConfig("Sald", "data_big/sald-dataset.bin", "data_big/sald-query.bin", 128, 1000,query_index_name=5),
]# embed_CONFIGS
# Selected datasets for training
SELECTED_DATASETS = [0, 1]
FINE_SELECTED_DATASETS = [0, 1]
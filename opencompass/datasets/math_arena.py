from datasets import load_dataset
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

@LOAD_DATASET.register_module()
class Hmmt2025Dataset(BaseDataset):

    @staticmethod
    def load():
        dataset = load_dataset('MathArena/hmmt_feb_2025')['train']
        return dataset

@LOAD_DATASET.register_module()
class Cmimc2025Dataset(BaseDataset):

    @staticmethod
    def load():
        dataset = load_dataset('MathArena/cmimc_2025')['train']
        return dataset

@LOAD_DATASET.register_module()
class Brumo2025Dataset(BaseDataset):

    @staticmethod
    def load():
        dataset = load_dataset('MathArena/brumo_2025')['train']
        return dataset
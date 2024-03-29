import os
from functools import partial

os.environ['TRANSFORMERS_CACHE'] = 'data/hg_data/transformers'
os.environ['HF_DATASETS_CACHE'] = 'data/hg_data/datasets'

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd


class Dataloader:

    def __init__(self, data_dir, max_length): 
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer_dict = {}
        self.necessary_items = None
        self.task_name = None
        self.dataset_info = None
        self.train_dataset_name = None
        self.load_func_dict = {
            'json': self._load_json_dataset,
            'tsv': self._load_tsv_dataset,
            'csv': self._load_csv_dataset,
        }

    def _load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer_dict[model_name] = tokenizer
        return tokenizer

    @staticmethod
    def _load_json_dataset(dataset_path):
        return load_dataset('json', data_files=dataset_path)['train']

    @staticmethod
    def _load_tsv_dataset(dataset_path):
        df = pd.read_csv(dataset_path, sep='\t')
        return Dataset.from_pandas(df.dropna(axis=0).reset_index(drop=True))
    
    @staticmethod
    def _load_csv_dataset(dataset_path):
        df = pd.read_csv(dataset_path)
        return Dataset.from_pandas(df.dropna(axis=0).reset_index(drop=True))

    def _tokenize_func(self, examples, tokenizer, max_length):
        text = [examples[text] for text in self.necessary_items]
        result = tokenizer(*text, padding=False, max_length=max_length, truncation=True)
        return result

    def _process(self, dataset, dataset_name, model_name):
        # rename the original name from the dataset to the standard name
        for item in [*self.necessary_items, 'labels', 'uid']:
            if item not in self.dataset_info[dataset_name]:
                continue
            if self.dataset_info[dataset_name][item] == item:
                continue
            dataset = dataset.rename_column(self.dataset_info[dataset_name][item], item)
        
        # add data indices 
        dataset = dataset.add_column('data_idx', list(range(len(dataset))))
        # convert string labels to class indices
        if isinstance(dataset[0]['labels'], str):
            dataset = dataset.map(lambda example: {
                'labels': self.dataset_info[dataset_name]['label_name_to_label'][example['labels']]})

        # remove additional columns
        add_info_columns = self.dataset_info[dataset_name]['additional_information'] \
            if 'additional_information' in self.dataset_info[dataset_name] else []
        extra_columns = [item for item in dataset.column_names if item not in
                         [*self.necessary_items, *add_info_columns, 'labels', 'data_idx', 'uid']]

        # process data
        dataset = dataset.map(partial(
            self._tokenize_func,
            tokenizer=self.tokenizer_dict[model_name],
            max_length=self.max_length),
            batched=True, remove_columns=extra_columns)

        return dataset

    def _load_dataset(self, dataset_name, split, model_name):
        dataset_path, dataset_type = self.dataset_info[dataset_name]['dataset_path_type'][split]
        dataset_path = os.path.join(self.data_dir, dataset_path)
        dataset = self.load_func_dict[dataset_type](dataset_path)
        if model_name not in self.tokenizer_dict:
            _ = self._load_tokenizer(model_name)
        dataset = self._process(dataset, dataset_name, model_name)
        add_info_columns = self.dataset_info[dataset_name]['additional_information'] \
            if 'additional_information' in self.dataset_info[dataset_name] else []
        forward_columns = set(dataset.column_names) - set([*self.necessary_items, *add_info_columns, 'uid'])
        info_columns = set(dataset.column_names) - forward_columns
        return dataset.remove_columns(list(info_columns)), dataset.remove_columns(list(forward_columns))

    def load_train(self, model_name='bert-base-uncased'):
        return self._load_dataset(self.train_dataset_name, 'train', model_name)

    def load_dev(self, dataset_names=None, model_name='bert-base-uncased'):
        if dataset_names is None:
            dataset_names = [dataset_name for dataset_name in self.dataset_info
                             if 'dev' in self.dataset_info[dataset_name]['dataset_path_type']]
        return {dataset_name: self._load_dataset(dataset_name, 'dev', model_name) for dataset_name in dataset_names}

    def load_test(self, dataset_names=None, model_name='bert-base-uncased'):
        if dataset_names is None:
            dataset_names = [dataset_name for dataset_name in self.dataset_info
                             if 'test' in self.dataset_info[dataset_name]['dataset_path_type']]
        return {dataset_name: self._load_dataset(dataset_name, 'test', model_name) for dataset_name in dataset_names}


class NLIDataloader(Dataloader):
    # required fields: premise, hypothesis, labels
    def __init__(self, data_dir='data/datasets/nli', max_length=None, train_dataset_name='mnli'):
        super().__init__(data_dir, max_length)
        self.train_dataset_name = train_dataset_name
        self.task_name = 'nli'
        self.necessary_items = ['premise', 'hypothesis']
        self.dataset_info = {
            'mnli': {
                'dataset_path_type': {
                    'train': ('mnli/train.tsv', 'mnli_tsv'),
                    'dev': ('mnli/dev_matched.tsv', 'mnli_tsv'),
                },
                'premise': 'sentence1',
                'hypothesis': 'sentence2',
                'labels': 'gold_label',
                'label_name_to_label': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
                'label_names': ['entailment', 'neutral', 'contradiction'],
                'uid': 'index', 
            },
            'anli_r1': {
                'dataset_path_type': {
                    'dev': ('anli/R1/dev.jsonl', 'json'), 
                    'test': ('anli/R1/test.jsonl', 'json')
                },
                'premise': 'context',
                'hypothesis': 'hypothesis',
                'labels': 'label',
                'label_name_to_label': {'e': 0, 'n': 1, 'c': 2},
                'label_names': ['entailment', 'neutral', 'contradiction'],
                'additional_information': ['genre', 'reason', 'model_label'],
            }, 
            'anli_r2': {
                'dataset_path_type': {
                    'dev': ('anli/R2/dev.jsonl', 'json'), 
                    'test': ('anli/R2/test.jsonl', 'json')
                },
                'premise': 'context',
                'hypothesis': 'hypothesis',
                'labels': 'label',
                'label_name_to_label': {'e': 0, 'n': 1, 'c': 2},
                'label_names': ['entailment', 'neutral', 'contradiction'],
                'additional_information': ['genre', 'reason', 'model_label'],
            }, 
            'anli_r3': {
                'dataset_path_type': {
                    'dev': ('anli/R3/dev.jsonl', 'json'), 
                    'test': ('anli/R3/test.jsonl', 'json')
                },
                'premise': 'context',
                'hypothesis': 'hypothesis',
                'labels': 'label',
                'label_name_to_label': {'e': 0, 'n': 1, 'c': 2},
                'label_names': ['entailment', 'neutral', 'contradiction'],
                'additional_information': ['genre', 'reason', 'model_label'],
            }, 
        }

        # Specify the loading function for each dataset
        self.load_func_dict['mnli_tsv'] = self._load_mnli_train_tsv_dataset

    @staticmethod
    def _load_mnli_train_tsv_dataset(dataset_path):
        df = {'sentence1': [], 'sentence2': [], 'gold_label': [], 'index': []}
        with open(dataset_path) as fin:
            fin.readline()
            for line in fin:
                line = line.strip().split('\t')
                df['index'].append(str(line[0]))
                df['sentence1'].append(line[8])
                df['sentence2'].append(line[9])
                df['gold_label'].append(line[-1])
        df = pd.DataFrame(df)
        return Dataset.from_pandas(df)


class HSDDataloader(Dataloader):
    # required fields: text, labels
    def __init__(self, data_dir='data/datasets/hsd', max_length=None, train_dataset_name='cad'):
        super().__init__(data_dir, max_length)
        self.train_dataset_name = train_dataset_name
        self.task_name = 'hsd'
        self.necessary_items = ['text']
        self.dataset_info = {
            'cad': {
                'dataset_path_type': {
                    'train': ('cad/cad_uid.train', 'hsd_tsv'),
                    'dev': ('cad/cad_uid.dev', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
                'uid': 'uid', 
            },
            'dynahate_r2_original': {
                'dataset_path_type': {
                    'dev': ('dynahate/r2_original.dev', 'hsd_tsv'),
                    'test': ('dynahate/r2_original.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r2_perturbation': {
                'dataset_path_type': {
                    'dev': ('dynahate/r2_perturbation.dev', 'hsd_tsv'),
                    'test': ('dynahate/r2_perturbation.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r3_original': {
                'dataset_path_type': {
                    'dev': ('dynahate/r3_original.dev', 'hsd_tsv'),
                    'test': ('dynahate/r3_original.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r3_perturbation': {
                'dataset_path_type': {
                    'dev': ('dynahate/r3_perturbation.dev', 'hsd_tsv'),
                    'test': ('dynahate/r3_perturbation.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r4_original': {
                'dataset_path_type': {
                    'dev': ('dynahate/r4_original.dev', 'hsd_tsv'),
                    'test': ('dynahate/r4_original.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r4_perturbation': {
                'dataset_path_type': {
                    'dev': ('dynahate/r4_perturbation.dev', 'hsd_tsv'),
                    'test': ('dynahate/r4_perturbation.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
        }
        self.load_func_dict['hsd_tsv'] = self._load_hsd_tsv

    @staticmethod
    def _load_hsd_tsv(dataset_path):
        data = pd.read_csv(dataset_path, sep='\t', names=['labels', 'text', 'uid'])
        return Dataset.from_pandas(data)


DATALOADER_DICT = {
    'nli': NLIDataloader,
    'hsd': HSDDataloader,
}
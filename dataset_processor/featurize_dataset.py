import json
import os
from utils.common import load_file,  normalize_gcdc_sub_corpus
from utils.data_specific import featurize_dataset,  get_permutated_sentence_pairs
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter
import random


#defines valid task that can be performed on the dataset
dataset_options = {
    "wsj": {
        "tasks" : ["sentence-ordering"],
    },
    "gcdc": {
        "tasks" : ['3-way-classification', 'minority-classification', 'sentence-ordering', 'sentence-score-prediction'],
        "sub_corpus" : ['All', 'Clinton', 'Enron', 'Yelp', 'Yahoo'],
    }
}

class GCDCFeaturizer:
    def __init__(self, args):
        if args.task not in dataset_options['gcdc']['tasks']:
            raise Exception('handler for %s task is not defined.' % (args.task))
        if args.sub_corpus.lower() not in [x.lower() for x in dataset_options['gcdc']['sub_corpus']]:
            raise Exception('%s is not part of GCDC corpus' % (args.sub_corpus))

        args.logger.info('featurizing the gcdc corpus: %s for task: %s with model architecture: %s' % (args.sub_corpus, args.task, args.arch))
        #populating the corpus list
        self.corpus_list = []
        if args.sub_corpus.lower() == 'all':
            self.corpus_list = dataset_options['gcdc']['sub_corpus'][1:]
        else:
            self.corpus_list = [normalize_gcdc_sub_corpus(args.sub_corpus)]
        #cache the cmdline arguments
        self.args = args
    
    def get_task_specific_labels(self, data_input_map):
        if self.args.task == 'sentence-score-prediction':
            annotator_aggreement = [data_input_map['label1'], data_input_map['label2'], data_input_map['label3']]
            annotator_aggreement = [x for x in map(lambda x: int(x), annotator_aggreement)]
            #regression task and it can be in range [0, 3]
            labels = np.mean(annotator_aggreement)
            return labels
        elif self.args.task == 'minority-classification':
            annotator_aggreement = [data_input_map['label1'], data_input_map['label2'], data_input_map['label3']]
            annotator_aggreement = [x for x in map(lambda x: int(x), annotator_aggreement)]
            cc = Counter(annotator_aggreement)
            label, freq = cc.most_common(1)[0]
            #binary classification task, where it takes value either zero (denotes non-low coherence text) or one (denotes low coherence text).
            labels=0 # not low coherence
            if label==1 and freq >= 2:
                labels=1 # low coherence
            return labels
        elif self.args.task == '3-way-classification':

            # its 3-class classification problem
            labels = int(data_input_map['label']) - 1    # actual gcdc labels are {1=low, 2=medium, 3=high}, transformed to [0=low, 1=medium, 2=high].  
            return labels
        else:
            raise Exception('%s task specific label processing is not defined' % self.args.task)
    
    def load_data(self, inference):
        dataset_map = {
            'train': [],
            'dev': [],
            'test': []
        }
        
        if inference:
            # removing training and development dataset while doing inference
            del dataset_map['dev']
            del dataset_map['train']
        else:
            del dataset_map['test']

        gcdc_dataset_path = os.path.join(self.args.processed_dataset_path, 'GCDC')
        self.args.logger.debug('<<LOADING>> GCDC dataset from directory: %s' % gcdc_dataset_path)
        for set_type in dataset_map:
            self.args.logger.debug('working on %s dataset ' % set_type)
            for sub_corpus in self.corpus_list:
                file_name = "%s_%s.jsonl"%(sub_corpus, set_type)
                input_data = load_file(os.path.join(gcdc_dataset_path, file_name))
                dataset_map[set_type].extend(input_data)
            self.args.logger.debug('<Done>')
        self.args.logger.debug('--'*30)
        return dataset_map

    def featurize_dataset(self, inference=False):
        
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        fact_tokenizer = tokenizer

        save_dir = os.path.join(self.args.processed_dataset_path, 'featurized_dataset')
        os.makedirs(save_dir, exist_ok=True)
        dataset_map = self.load_data(inference)
        # include_facts = (self.args.arch in ['fact-aware', 'combined']) 
        sent_sep = tokenizer.sep_token if (self.args.arch in ['hierarchical'])  else ' '

        # task specific handler
        if self.args.task in ['3-way-classification', 'minority-classification', 'sentence-score-prediction']: 
            for set_type, data_list in dataset_map.items():
                for data_instance in data_list:
                    if self.args.arch in ['hierarchical'] :
                        data_instance['doc_a'] = sent_sep.join([' '.join(x) for x in data_instance['sentences']])
                    else:
                        data_instance['doc_a'] = data_instance['text']
                    data_instance['label'] = self.get_task_specific_labels(data_instance)


        # pairwise sentence ordering task handler 
        if self.args.task == "sentence-ordering":
            for set_type, data_list in dataset_map.items():
                
                permutated_pair_data_list = get_permutated_sentence_pairs(data_list, self.args.permutation_count, 
                                                    self.args.logger, with_replacement=self.args.with_replacement>0,
                                                    sentence_separator=sent_sep, inverse=self.args.inverse_pra>0)
                #update the dataset with permutated sentence pair
                dataset_map[set_type] = permutated_pair_data_list

        # featurizing the dataset
        for set_type, data_list in dataset_map.items():
            save_file_path = os.path.join(save_dir, '%s.jsonl'%set_type)
            featurize_dataset(tokenizer, fact_tokenizer,  self.args.max_fact_count, self.args.max_seq_len, 
                                            self.args.max_fact_seq_len, save_file_path, data_list, self.args.logger)
        self.args.logger.debug('--'*30)
        
        # label distribution
        if self.args.task != 'sentence-score-prediction':
            for set_type, data_list in dataset_map.items():
                self.args.logger.info('label distribution in %s dataset (total count: %d)' % (set_type, len(data_list)))
                lc = Counter([x['label'] for x in data_list])
                self.args.logger.info('%s' % ({k:v for k,v in lc.items()}))
            self.args.logger.debug('--'*30)


class WSJFeaturizer:
    def __init__(self, args):
        if args.task not in dataset_options['wsj']['tasks']:
            raise Exception('handler for %s task is not defined.' % (args.task))

        args.logger.info('featurizing the wsj corpus for task: %s with model architecture: %s' % (args.task, args.arch))
        #cache the cmdline arguments
        self.args = args
    
    def load_data(self, inference):
        dataset_map = {
            'train': [],
            'dev': [],
            'test': []
        }

        if inference:
            # removing training and development dataset while doing inference
            del dataset_map['dev']
            del dataset_map['train']
        else:
            del dataset_map['test']

        wsj_dataset_path = os.path.join(self.args.processed_dataset_path, 'WSJ')
        self.args.logger.debug('<<LOADING>> WSJ dataset from directory: %s' % wsj_dataset_path)
        for set_type in dataset_map:
            self.args.logger.debug('working on %s dataset ' % set_type)
            file_name = "%s.jsonl"%(set_type)
            input_data = load_file(os.path.join(wsj_dataset_path, file_name))
            dataset_map[set_type] = input_data
            self.args.logger.debug('<Done>')
        self.args.logger.debug('--'*30)
        return dataset_map

    def featurize_dataset(self, inference=False):
        # handling special ablation study cases

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        fact_tokenizer = tokenizer

        save_dir = os.path.join(self.args.processed_dataset_path, 'featurized_dataset')
        os.makedirs(save_dir, exist_ok=True)
        dataset_map = self.load_data(inference)

        sent_sep=' '

        # integrating facts if required
        # pairwise sentence ordering task handler 
        for set_type, data_list in dataset_map.items(): 
            permutated_pair_data_list = get_permutated_sentence_pairs(data_list,  self.args.permutation_count, 
                                                self.args.logger, with_replacement=self.args.with_replacement>0,
                                                sentence_separator=sent_sep, inverse=self.args.inverse_pra>0)
            #update the dataset with permutated sentence pair
            dataset_map[set_type] = permutated_pair_data_list

        # featurizing the dataset
        for set_type, data_list in dataset_map.items():
            save_file_path = os.path.join(save_dir, '%s.jsonl'%set_type)
            featurize_dataset(tokenizer, fact_tokenizer, self.args.max_fact_count, self.args.max_seq_len, 
                                            self.args.max_fact_seq_len, save_file_path, data_list, self.args.logger)
        self.args.logger.debug('--'*30)
        
        # label distribution
        for set_type, data_list in dataset_map.items():
            self.args.logger.info('label distribution in %s dataset (total count: %d)' % (set_type, len(data_list)))
            lc = Counter([x['label'] for x in data_list])
            self.args.logger.info('%s' % ({k:v for k,v in lc.items()}))
        self.args.logger.debug('--'*30)


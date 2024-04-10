import numpy as np
import itertools
from .common import random_seed, linearize_facts, store_results, encode_batch
from tqdm import tqdm
import random

def get_permutations(sentence_count, permutation_count):
    res = []
    if sentence_count < 6:
        total_count = 0
        original_order = [x for x in range(sentence_count)]
        perms = set(itertools.permutations(original_order))
        for perm_order in perms:
            perm_order = list(perm_order)
            if np.all(np.array(perm_order)==np.array(original_order)):
                continue
            if total_count >= permutation_count:
                break
            total_count+=1
            res.append(perm_order)
    else:
        original_order = np.array([x for x in range(sentence_count)])
        prev_perms = []
        for j in range(permutation_count):
            perm_order = np.random.permutation(sentence_count)
            perm_str = ','.join([str(z) for z in perm_order])
            while np.all(perm_order==original_order) or perm_str in prev_perms:
                perm_order = np.random.permutation(sentence_count)
                perm_str = ','.join([str(z) for z in perm_order])
            prev_perms.append(perm_str)
            res.append(perm_order)
    random.shuffle(res)
    return res

def get_permutations_with_replacement(sentence_count, permutation_count):
    res = []
    original_order = np.array([x for x in range(sentence_count)])
    for j in range(permutation_count):
        perm_order = np.random.permutation(sentence_count)
    
        while np.all(perm_order==original_order):
            perm_order = np.random.permutation(sentence_count)
        res.append(perm_order)
    return res

def get_permutated_sentence_pairs(data_list,  permutations, logger, seed=42, with_replacement=True, inverse=False, sentence_separator=' '): 
    random_seed(seed)
    res = []
    one_sent_doc_count = 0
    total_count = 0

    for data in tqdm(data_list):
        total_count += 1
        #remove documents containing only one sentence
        sentences = data['sentences']
        
        if len(sentences)<=1:
            one_sent_doc_count+=1
            continue
        

        if with_replacement:
            permutation_ordering = get_permutations_with_replacement(len(sentences), permutations)
        else:
            permutation_ordering = get_permutations(len(sentences), permutations)

        if inverse:
            permutation_ordering = [[len(sentences)-1-j for j in range(len(sentences))]]

        doc_1 = sentence_separator.join([' '.join(x) for x in sentences])
        
        
        for idx, perm_order in enumerate(permutation_ordering):
            doc_2 = sentence_separator.join([' '.join(sentences[i]) for i in perm_order])
            
            #create balanced dataset
            json_data = {
                'doc_a': doc_1,
                'doc_b': doc_2,
                'label': -1,
            }


            if idx%2==0:
                json_data = {
                    'doc_a': doc_2,
                    'doc_b': doc_1,
                    'label': 1,
                }

            res.append(json_data)
    logger.info('single sentence doc count: %d, total doc count: %d, obtained permutation dataset count: %d' % (one_sent_doc_count, total_count, len(res)))
    return res


def process_facts(facts):
    def is_valid_triple(fact_triples):
        status = True
        for y in fact_triples:
            status &= (y.strip()!='' and len(y.strip())!=0)
        return status

    processed_facts = []
    for x in facts:
        if len(x)==3 and is_valid_triple(x):
            processed_facts.append(x)
    return processed_facts 
    
def process_sentences_and_facts(sents, facts, max_sequence_length):
    processed_sents = []
    processed_facts = []
    word_length = 0
    is_truncated = False

    assert len(sents) == len(facts), "mismatch in number of sentences: %d and facts: %d." % (len(sents), len(facts))
    doc_fact_count = 0

    for x, x_facts in zip(sents, facts):
        words = x
        if max_sequence_length > 0 and word_length + len(words) > max_sequence_length:
            is_truncated = True
            break
        word_length += len(words)
        processed_sents.append(x)
        p_facts = process_facts(x_facts)
        doc_fact_count+=len(p_facts)
        processed_facts.append(p_facts)

    return processed_sents, processed_facts, is_truncated, doc_fact_count==0

def featurize_data(tokenizer, fact_tokenizer,  max_fact_count, max_seq_len, max_fact_seq_len, dataset, logger, exact_count=False):
    max_fact_count_per_doc = 0
    
    position_data = []
    concatenate_all_sentences = []
    logger.info('post-processing the dataset')
    for doc in tqdm(dataset):
        temp_data = {}

        #label information
        temp_data['label'] = doc['label']

        #concatenate sentences
        temp_data['doc_a_offset'] = len(concatenate_all_sentences)
        concatenate_all_sentences.append(doc['doc_a'])
        if 'doc_b' in doc:
            temp_data['doc_b_offset'] = len(concatenate_all_sentences)
            concatenate_all_sentences.append(doc['doc_b'])
        
        
        position_data.append(temp_data)
    
    if exact_count:
        max_fact_count_per_doc = max_fact_count
    
    logger.info('data preprocessed: %d, sentences preprocessed: %d' % (len(position_data), len(concatenate_all_sentences)))
    

    sentences_input_ids, sentences_attention_ids = encode_batch(tokenizer, concatenate_all_sentences, max_seq_len)
    
    max_sent_seq_len = len(sentences_input_ids[0])

    res_data = []
    logger.info('featurizing the datasets..')
    
    for pos in tqdm(position_data):
        temp = {}
        #label information
        temp['label'] = pos['label']
        #process the sentences
        temp['doc_a'] = sentences_input_ids[pos['doc_a_offset']]
        temp['doc_a_mask'] = sentences_attention_ids[pos['doc_a_offset']]
        
        if 'doc_b_offset' in pos and 'doc_b_offset' in pos:
            temp['doc_b'] = sentences_input_ids[pos['doc_b_offset']]
            temp['doc_b_mask'] = sentences_attention_ids[pos['doc_b_offset']]
        
        #process the facts
        
        res_data.append(temp)
    
    logger.debug('%d data instance processed. max sent_seq_length: %d' % (len(position_data), max_sent_seq_len))
    
    return res_data

def featurize_dataset(tokenizer, fact_tokenizer,  max_fact_count, max_seq_len, max_fact_seq_len, savefile, dataset, logger):
    res_data = featurize_data(tokenizer, fact_tokenizer,  max_fact_count, max_seq_len, max_fact_seq_len, dataset, logger)
    store_results(savefile, res_data)
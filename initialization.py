# -*- coding: utf-8 -*-
# code warrior: Barid
##########
import tensorflow as tf
import os, sys
from UNIVERSAL.MLM import MLM_base
from UNIVERSAL.basic_metric import mean_metric


cwd = os.getcwd()
domain_path = [            "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/EnDeHi_6k_wiki/EnDeHi_codes_6K.de.vocab.freq",
                       "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/EnDeHi_6k_wiki/EnDeHi_codes_6K.en.vocab.freq",
                               "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/EnDeHi_6k_wiki/EnDeHi_codes_6K.hi.vocab.freq",]
def get_domain_index():
    sorted_domain_list = []
    for domain in domain_path:
        domain_list = dict()
        with open(
            domain,
            "r",
        ) as f:
            for k, v in enumerate(f.readlines()):
                _, ids, c = v.strip().split(" ")
                domain_list[int(ids)] = float(c)
        domain_list = sorted(domain_list.items(), key=lambda x: x[1], reverse=True)
        sorted_domain_list.append([d[0] for d in domain_list])
    return sorted_domain_list


en_sorted, de_sorted, hi_sorted = get_domain_index()
lang_sorted = tf.constant(tf.keras.utils.pad_sequences([en_sorted, de_sorted, hi_sorted],padding="post"))
en_sorted_freq = en_sorted[:20000]
de_sorted_freq = de_sorted[:20000]
hi_sorted_freq = hi_sorted[:20000]
lang_sorted_freq = tf.constant([en_sorted_freq,de_sorted_freq,hi_sorted_freq])
en_sorted_len = len(en_sorted)
de_sorted_len = len(de_sorted)
hi_sorted_len = len(hi_sorted)
lang_sorted_len = tf.constant([en_sorted_len, de_sorted_len, hi_sorted_len])
en_unique = set(lang_sorted_freq[0].numpy()).difference(set(lang_sorted_freq[1].numpy())).difference(set(lang_sorted_freq[2].numpy()))
de_unique = set(lang_sorted_freq[1].numpy()).difference(set(lang_sorted_freq[2].numpy())).difference(set(lang_sorted_freq[0].numpy()))
hi_unique = set(lang_sorted_freq[2].numpy()).difference(set(lang_sorted_freq[0].numpy())).difference(set(lang_sorted_freq[1].numpy()))
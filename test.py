#!/usr/bin/env python3.6
#-*-coding=utf-8-*-

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def build_vocab(
    sequences,
    token_to_idx={},
    min_token_count=1,
    delim=' ',
    punct_to_keep=None,
    punct_to_remove=None,
):
    print("sequences = {}".format(sequences))
    print("token_to_idx = {}".format(token_to_idx))
    print("token_to_idx = {}".format(len(token_to_idx)))
    print("min_token_count = {}".format(min_token_count))
    print("delim = {}".format(delim))
    print("punct_to_keep = {}".format(punct_to_keep))
    print("punct_to_remove = {}".format(punct_to_remove))
    return None


sentences1 = ['hello, jack.']
    
build_vocab(sentences1, SPECIAL_TOKENS, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
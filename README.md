# Tokenizer from Scratch

A tokenizer built using the Byte-Pair Encoding (BPE) algorithm. Can be used to train a tokenizer using text in any language. Then, using the trained tokenizer, you can tokenize an LLM training dataset.

## Functionality

Builds a `vocab` dict that has ID keys (the token) and a values of the what the token represent (a byte, char, subword, or word).

Initially, it starts filling the vocab with tokens for all values in a single byte (`0-255`). Then training data is encoded using UTF-8 encoding (each possible character has 1-4 byte representation). The BPE algorithm finds the most freuqent byte pairs, merges them into a new token and adds them to `vocab`.  This process is repeated untill `len(vocab) == max_vocab`.
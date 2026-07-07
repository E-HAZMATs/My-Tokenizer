'''
The tokenizer takes text > encodes to utf8 > merges most frequent byte merges >
pair becomes new token > added to vocab > repeat till max vocab size.

Tokenizer also handles encoding and decoding.
'''

'''
encode text -> add vocab _ replace encoded text by finding toeknized merges and replacing that with new token > repeat

- confused about vocab, I guess it should be a dict. Key = byte or byte pair (but key as byte pair would get so long tho? say one token is the result of merge 10 times, ti's key will be huge?)
'''

'''
GENERAL TODOS:
TODO: Handle equality of freq occurrence? maybe no need, since the other pair will get selected next iter?
TODO: Try on Arabic dataset. Drop taylor swify txt.
TODO: Serialize the tokenizer?
TODO: Export vocab and merges?
'''
class Tokenizer:
    
    def __init__(self):

        # The tokenizer should be able to generalize and encode tokens not in the training set.
        # Each possible character can be represented into 1-4 bytes in UTF-8. So if ت for example is not in training set,
        # then its utf8 representation, say for example it's [50, 200], will encode the letter as the 50 and 200 byte tokens.
        # Of course this is would cause a problem cus the GPT model's embedding for those bytes would probably be less learnt.
        # TODO: What about decoding tho? how would the tokenizer decode back to `ت`?
        # A: The bytes the way they're designed in utf8 sorta gives hint if whether the byte is a sibling to another byte(s).
        # so if the unkown char is [50,200] then the byte 50 here would somehow give an idea that it should concated with other byte(s) to get a char.
        self.vocab = {i: bytes([i]) for i in range(256)} # Base vocab. Should be {ID: Bytes}
        self.merges = {} # The token merges.  
        pass

    # max_vocab is the stopping point. Once the we have max_vocab tokens we stop merging.
    def train(self, dataset, max_vocab):
        # WORKFLOW: ENCODED DATA > FIND POPULAR PAIR > ADD NEW TOKEN > UPDATE DATASET, REPLACING PAIRS WITH NEW TOKEN > REPEAT TILL STOP POINT. 
        chars = sorted(list(set(dataset)))
        # assert len(chars) < max_vocab, "Dataset has more base tokens than `max_vocab`"
        assert max_vocab > 255, "can't have max_vocab less than length of base vocab." # should be 256?
        pairing_iter = max_vocab - 256 # could be 0... I guess that's fine
        data_enc = dataset.encode('utf-8')

        
        for i in range(pairing_iter):
            new_id = i + 256 # base is 0-255. So 1st next token in vocab is 256.

            # constructing a list of the keys alone.
            # freq_list = list(self._occur_freq(data_enc)) # There is a better way with `max` (no list construction). forgot how tho.
            freq_list = list(self._occur_freq(data_enc)) # There is a better way with `max` (no list construction). forgot how tho.
            top_pair_key = freq_list[0]
            data_enc = self.merge(data_enc, new_id, top_pair_key)
            self.vocab[new_id] = self.merges[top_pair_key] # Value already set by the `merge` method.

        
        print('*** FINISHED BUILDING VOCAB ***')

    def encode(self, text):
        '''
        How?
        maybe encode to utf8 > iteratively apply similar logic to `merge`?
        
        
        '''
        raise NotImplementedError

    def decode(self, token):
        raise NotImplementedError

    # Recieves the byte representation of the text, returns the most occurrent pair?
    # CHECK: what if equality?    
    def _occur_freq(self, data):
        freqs = dict()

        # When data reaches last el, `zip` wouldn't produce a pair cus `data[1:]` is out of bound.
        for pair in zip(data, data[1:]):
            # If key exists, return its value after adding 1, else return 0(fallback) + 1.
            freqs[pair] = freqs.get(pair, 0) + 1
        
        # TODO?: Should just return the max pair? Handle equality of occurrence?
        freqs_sorted = dict(sorted(freqs.items(), key= lambda item: -item[1]))
        return freqs_sorted

    # "Edits" the training data to use new token, update `merges` dict.
    def merge(self, data_enc, new_id, pair):
        data_updated = []

        # CHECK THIS. Key is the merge pair, value is their bytes concated.
        self.merges[pair] = self.vocab[pair[0]] + self.vocab[pair[1]]
        i = 0
        while i < len(data_enc):
            
            # 2nd cond handles if last token is same as pair[0], i+1 would go out of bound and throw.
            # can't keep it outside cus if last 2 tokens are of `pair` then i would increment by 2 and go out of bound.
            if data_enc[i] == pair[0] and i < len(data_enc)-1 and data_enc[i+1] == pair[1]:
                data_updated.append(new_id)
                i += 2
            else:
                data_updated.append(data_enc[i])
                i += 1

        return data_updated

# -------
# CONFIGS / ARGS
max_vocab = 400
# -------


with open('taylorswift.txt') as f:
    data = f.read()

tokenizer = Tokenizer()

tokenizer.train(data, max_vocab)

from pprint import pprint
pprint(tokenizer.vocab)
print('*******')
print('*******')
pprint(tokenizer.merges) 
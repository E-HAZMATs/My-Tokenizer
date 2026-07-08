'''
The tokenizer takes text > encodes to utf8 > merges most frequent byte merges >
pair becomes new token > added to vocab > repeat till max vocab size.

Tokenizer also handles encoding and decoding.
'''

'''
encode text -> add vocab _ replace encoded text by finding toeknized merges and replacing that with new token > repeat
'''

'''
GENERAL TODOS:
TODO: Handle equality of freq occurrence? maybe no need, since the other pair will get selected next iter?
TODO: Try on Arabic dataset. Drop taylor swift txt.
TODO: Serialize the tokenizer?
TODO: Export vocab and merges?
'''
from os import path
from pprint import pprint
import json
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
        assert max_vocab > 255, "can't have max_vocab less than length of base vocab."
        pairing_iter = max_vocab - 256 # could be 0... I guess that's fine
        data_enc = dataset.encode('utf-8')
        
        for i in range(pairing_iter):
            new_id = i + 256 # base is 0-255. So 1st next token in vocab is 256.

            # constructing a list of the keys alone.
            freq_list = list(self._occur_freq(data_enc)) # There is a better way with `max` (no list construction). forgot how tho.
            top_pair = freq_list[0]
            data_enc = self.merge(data_enc, new_id, top_pair)
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

        
        print('*** FINISHED BUILDING VOCAB ***')

    def encode(self, text):
        assert True # Add assertion that checks if tokenizer is trained? if not trained it will just return utf8 no?
        text_enc = text.encode('utf-8')
        # In case text is just 1 char, no merging needed.
        i = 0
        while len(text_enc) > 1:

            freqs = self._occur_freq(text_enc)
            '''
            iterates over `freqs` keys > if the key is in `merges` we get that value, else we get infinity > we compare the values and find min in which it's the first merge for the given pair.
            `inf` = not in `merges`.
            '''
            pair = min(freqs, key= lambda pair: self.merges.get(pair, float('inf')))

            # Should be true only after getting the final merge done in training.
            if pair not in self.merges:
                break 
            i += 1
            if i == 40:
                break
            new_token = self.merges[pair]
            text_enc = self.merge(text_enc, new_token, pair, encoding=True)
        return text_enc


    def decode(self, tok_seq):
        # Must concatenate byte sequence cus 1 char could 1-4 bytes. 
        decoded = b''.join([self.vocab[tok] for tok in tok_seq]).decode('utf-8')
        return decoded

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
    def merge(self, data_enc, new_id, pair, encoding=False):
        data_updated = []

        # When encoding don't update `merges` dict. Better to outsource this op tho.
        if not encoding:
            self.merges[pair] = new_id # Connects this dict with `vocab`. Value could [new_id, new byte(s)] instead?

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

    def save(self):
        voc = self.vocab.copy()
        # BPE works on byte rep of the text, so it could pair 2 bytes that correspond to nothing in UTF-8 and can't be decoded.
        # Use errors=replace stops it from throwing and replaces rubbish byte pairs with some symbol.
        voc = {k: v.decode('utf-8', errors='replace') for k, v in voc.items()}
        with open('checkpoints/vocab.json', 'w') as f:
            json.dump(self.vocab, f, indent=4)
        
# -------
# CONFIGS / ARGS
max_vocab = 280
data_path = 'data'
data_set = ['ww1-wiki-ar.txt', 'taylorswift.txt']

test_text = '''For dicts to be ordered, you need Python 3.7+, or 3.6+ if you're okay with relying on the technically-an-implementation-detail ordered nature of dicts on CPython 3.6.'''

# test_text = '''توقَّف التقدم الألماني في فرنسا في معركة المارن، وبحلول نهاية عام 1914 استقرت الجبهة الغربية على حرب استنزاف تميزت بسلسلة طويلة من خطوط الخنادق التي قليلاً ما تغيَّرت حتى عام 1917. على الجبهة الشرقية دخل جيشان روسيان شرق بروسيا في 17 أغسطس بناءً للاتفاق مع فرنسا عام 1912 بمهاجمة ألمانيا خلال 15 يومًا من التعبئة. أجبر الألمان على تحويل قوات من الغرب، لكنهم نجحوا في صدِّ هذا الغزو بانتصارٍ في معركة تاننبرغ ومعركة بحيرات ماسوريان الأولى، ومع ذلك احتل الروس مقاطعة غاليسيا الشرقية في النمسا-المجر.'''
# -------



with open(path.join(data_path, data_set[1]), encoding='utf-8') as f:
    data = f.read()

tokenizer = Tokenizer()

tokenizer.train(data, max_vocab)


pprint(tokenizer.vocab)
print('*******')
print('*******')
# pprint(tokenizer.merges) 
print('\n\n')
text_enc = tokenizer.encode(test_text)
print(len(text_enc))
print('\n\n')
text_dec = tokenizer.decode(text_enc)
print(text_dec)
print(text_dec == test_text)

tokenizer.save()
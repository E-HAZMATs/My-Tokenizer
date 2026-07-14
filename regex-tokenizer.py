import regex as re
from os import path

'''
TODO: Add special chars?
'''

# GPT2 Regex but I added handling for tashkeel/harakat. They're under category `Mn`.
# The regex would match each letter with a following zero or more `tashkeel`.
# Example: "توقَّف" would be a single chunk with this regex instead of being split when the re hits the diacritics.
GPT2_SPLIT_PATTERN_DIACRITICS = r"""'(?:[sdmt]|ll|ve|re)| ?(?:\p{L}\p{Mn}*)+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class RegexTokenizer:
    
    def __init__(self):
        self.reg = re.compile(GPT2_SPLIT_PATTERN_DIACRITICS)
        self.vocab = {i: bytes([i]) for i in range(256)} # Base vocab. Should be {ID: Bytes}
        self.merges = {} # The token merges.  
        self.special_toks = {}
        self.special_toks_inv = {}

    def train(self, dataset, max_vocab):
        assert max_vocab > 255, "can't have max_vocab less than length of base vocab."
        pairing_iter = max_vocab - 256
        chunks = re.findall(self.reg, dataset) # Split
        # UTF-8 enocdes each chunk in the training data.
        # A list of byte lists. Each byte list is processes separately when merging.
        data_enc = [list(ch.encode('utf-8')) for ch in chunks]
        for i in range(pairing_iter):
            new_id = i + 256 # base is 0-255. So 1st next token in vocab is 256.

            # FIXME: CHANGED `occur_freq` to get a chunk. This could break REVIEW!!!!!
            freqs = dict()
            for chunk in data_enc:
                freqs = self._occur_freq(chunk, freqs)
            
            # sort dict then construct a list of it to get more frequent pair.
            # outsourced from `_occur_freq` since the method is used more broadly now.
            freqs = list(dict(sorted(freqs.items(), key= lambda item: -item[1])))
            # constructing a list of the keys alone.
            top_pair = freqs[0]
            self.merges[top_pair] = new_id # Connects this dict with `vocab`. Value could [new_id, new byte(s)] instead?
            data_enc = [self.merge(chunk, new_id, top_pair) for chunk in data_enc]
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

        with open('reg_vocab.txt', 'w', encoding='utf-8') as f:
            for k, v in self.vocab.items():
                f.write(f'{k}: {v.decode("utf-8", errors="replace")}\n')
    
    def _occur_freq(self, data, freqs=None):
        freqs = dict() if freqs is None else freqs
        # When data reaches last el, `zip` wouldn't produce a pair cus `data[1:]` is out of bound.
        for pair in zip(data, data[1:]):
            # If key exists, return its value after adding 1, else return 0(fallback) + 1.
            freqs[pair] = freqs.get(pair, 0) + 1
        return freqs
        # return list(freqs_sorted)

    # "Edits" the training data to use new token, update `merges` dict.
    def merge(self, data_enc, new_id, pair, encoding=False): # XXX: `encoding` is useless here. I outsourced merging so... delete it?
        data_updated = []
        i = 0
        while i < len(data_enc):
            if data_enc[i] == pair[0] and i < len(data_enc)-1 and data_enc[i+1] == pair[1]:
                data_updated.append(new_id)
                i += 2
            else:
                data_updated.append(data_enc[i])
                i += 1

        return data_updated
    
    def encode_chunk(self, chunk):
        while len(chunk) > 1:
            freqs = self._occur_freq(chunk)
            pair = min(freqs, key= lambda pair: self.merges.get(pair, float('inf')))
            if pair not in self.merges:
                break
            new_token = self.merges[pair]
            chunk = self.merge(chunk, new_token, pair)
        return chunk

    def encode(self, text):
        chunks = re.findall(self.reg, text)
        chunks_enc = [list(chunk.encode('utf-8')) for chunk in chunks]
        enc = []
        for chunk in chunks_enc:
            enc.extend(self.encode_chunk(chunk))
        
        return enc

    
    def decode(self, tok_seq):
        decoded = b''.join([self.vocab[tok] for tok in tok_seq]).decode('utf-8')
        return decoded 

    def add_special_toks(self, special):
        assert isinstance(special, dict) 
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in special.items())
        self.special_toks = special
        self.special_toks_inv = {v: k for k, v in special.items()}
    
    def encode_spec(self, text):

        # str for a regex. we escape each special char in the special toks.
        # So '|' become "/|". This way, regex doesn't count it as "or".
        # parenthesis on each end makes the findall return the matched pattern with the list 
        special_pattern = '(' + '|'.join(re.escape(k) for k in self.special_toks) + ')'
        # splitting text when hitting a special char.
        # "Hello<|endoftext|>there" becomes ['hello', '<|endoftext|>', 'there]
        special_chunks = re.split(special_pattern, text)

        toks = []
        for chunk in special_chunks:
            if chunk in self.special_toks:
                toks.append(self.special_toks[chunk])
            else:
                toks.extend(self.encode(chunk))
        
        return toks


    # -------
# CONFIGS / ARGS
max_vocab = 400
data_path = 'data'
data_set = ['ww1-wiki-ar.txt', 'taylorswift.txt']

# test_text = '''For dicts to be ordered, you need Python 3.7+, or 3.6+ if you're okay with relying on the technically-an-implementation-detail ordered nature of dicts on CPython 3.6.'''

test_text = '''توقَّف التقدم الألماني في فرنسا في معركة المارن، وبحلول نهاية عام 1914 استقرت الجبهة الغربية على حرب استنزاف تميزت بسلسلة طويلة من خطوط الخنادق التي قليلاً ما تغيَّرت حتى عام 1917. على الجبهة الشرقية دخل جيشان روسيان شرق بروسيا في 17 أغسطس بناءً للاتفاق مع فرنسا عام 1912 بمهاجمة ألمانيا خلال 15 يومًا من التعبئة. أجبر الألمان على تحويل قوات من الغرب، لكنهم نجحوا في صدِّ هذا الغزو بانتصارٍ في معركة تاننبرغ ومعركة بحيرات ماسوريان الأولى، ومع ذلك احتل الروس مقاطعة غاليسيا الشرقية في النمسا-المجر.'''
# -------



with open(path.join(data_path, data_set[1]), encoding='utf-8') as f:
    data = f.read()

tokenizer = RegexTokenizer()

tokenizer.train(data, max_vocab)
tokenizer.add_special_toks({"<|endoftext|>": max_vocab})
h = 'Hello there.<|endoftext|>I love tuna.'
# h = 'م'
# text_enc = tokenizer.encode(h)
text_enc = tokenizer.encode_spec(h)
print(text_enc)
# text_dec = tokenizer.decode(text_enc)
# print(test_text == text_dec)

# print(text_dec)
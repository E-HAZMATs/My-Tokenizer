import regex as re
from os import path


# GPT2 Regex but I added handling for tashkeel/harakat. They're under category `Mn`.
# The regex would match each letter with a following zero or more `tashkeel`.
# Example: "توقَّف" would be a single chunk with this regex instead of being split when the re hits the diacritics.
GPT2_SPLIT_PATTERN_DIACRITICS = r"""'(?:[sdmt]|ll|ve|re)| ?(?:\p{L}\p{Mn}*)+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class RegexTokenizer:
    
    def __init__(self):
        self.reg = re.compile(GPT2_SPLIT_PATTERN_DIACRITICS)
        self.vocab = {i: bytes([i]) for i in range(256)} # Base vocab. Should be {ID: Bytes}
        self.merges = {} # The token merges.  

    def train(self, dataset, max_vocab):
        chars = sorted(list(set(dataset)))
        assert max_vocab > 255, "can't have max_vocab less than length of base vocab."
        pairing_iter = max_vocab - 256
        chunks = re.findall(self.reg, dataset) # Split
        # UTF-8 enocdes each chunk in the training data.
        # A list of byte lists. Each byte list is processes separately when merging.
        data_enc = [list(ch.encode('utf-8')) for ch in chunks]
        for i in range(pairing_iter):
            new_id = i + 256 # base is 0-255. So 1st next token in vocab is 256.

            freqs = self._occur_freq(data_enc)

            # constructing a list of the keys alone.
            top_pair = freqs[0]
            self.merges[top_pair] = new_id # Connects this dict with `vocab`. Value could [new_id, new byte(s)] instead?
            data_enc = [self.merge(chunk, new_id, top_pair) for chunk in data_enc]
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

        with open('reg_vocab.txt', 'w', encoding='utf-8') as f:
            for k, v in self.vocab.items():
                f.write(f'{k}: {v.decode("utf-8", errors="replace")}\n')

    def _occur_freq(self, data):
        freqs = dict()
        for chunk in data:
            # When data reaches last el, `zip` wouldn't produce a pair cus `data[1:]` is out of bound.
            for pair in zip(chunk, chunk[1:]):
                # If key exists, return its value after adding 1, else return 0(fallback) + 1.
                freqs[pair] = freqs.get(pair, 0) + 1
        freqs_sorted = dict(sorted(freqs.items(), key= lambda item: -item[1]))
        return list(freqs_sorted)
    # "Edits" the training data to use new token, update `merges` dict.
    def merge(self, data_enc, new_id, pair, encoding=False):
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
    

    def encode(self, text):
        chunks = re.findall(self.reg, text)
        chunks_enc = [list(chunk.encode('utf-8')) for chunk in chunks]
        # In case text is just 1 char, no merging needed.
        while len(text) > 1:
            
            freqs = self._occur_freq(chunks_enc)
            '''
            iterates over `freqs` keys > if the key is in `merges` we get that value, else we get infinity > we compare the values and find min in which it's the first merge for the given pair.
            `inf` = not in `merges`.
            '''
            pair = min(freqs, key= lambda pair: self.merges.get(pair, float('inf')))

            # Should be true only after getting the final merge done in training.
            if pair not in self.merges:
                break 
            new_token = self.merges[pair]
            for i in range(len(chunks_enc)):
                chunks_enc[i] = self.merge(chunks_enc[i], new_token, pair, encoding=True)
        return chunks_enc # XXX: Should chunks be flattened?


    # -------
# CONFIGS / ARGS
max_vocab = 500
data_path = 'data'
data_set = ['ww1-wiki-ar.txt', 'taylorswift.txt']

# test_text = '''For dicts to be ordered, you need Python 3.7+, or 3.6+ if you're okay with relying on the technically-an-implementation-detail ordered nature of dicts on CPython 3.6.'''

test_text = '''توقَّف التقدم الألماني في فرنسا في معركة المارن، وبحلول نهاية عام 1914 استقرت الجبهة الغربية على حرب استنزاف تميزت بسلسلة طويلة من خطوط الخنادق التي قليلاً ما تغيَّرت حتى عام 1917. على الجبهة الشرقية دخل جيشان روسيان شرق بروسيا في 17 أغسطس بناءً للاتفاق مع فرنسا عام 1912 بمهاجمة ألمانيا خلال 15 يومًا من التعبئة. أجبر الألمان على تحويل قوات من الغرب، لكنهم نجحوا في صدِّ هذا الغزو بانتصارٍ في معركة تاننبرغ ومعركة بحيرات ماسوريان الأولى، ومع ذلك احتل الروس مقاطعة غاليسيا الشرقية في النمسا-المجر.'''
# -------



with open(path.join(data_path, data_set[0]), encoding='utf-8') as f:
    data = f.read()

tokenizer = RegexTokenizer()

tokenizer.train(data, max_vocab)

text_enc = tokenizer.encode(test_text)
print(text_enc)
# text_dec = tokenizer.decode(text_enc)
# tokenizer.save()
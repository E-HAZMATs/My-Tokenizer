'''
The tokenizer takes text > encodes to utf8 > merges most frequent byte pairs >
pair becomes new token > added to vocab > repeat till max vocab size.

Tokenizer also handles encoding and decoding.
'''

'''
encode text -> add vocab _ replace encoded text by finding toeknized pairs and replacing that with new token > repeat

- confused about vocab, I guess it should be a dict. Key = byte or byte pair (but key as byte pair would get so long tho? say one token is the result of merge 10 times, ti's key will be huge?)
'''
class Tokenizer:
    
    def __init__(self):
        self.vocab = None
        pass

    # max_vocab is the stopping point. Once the we have max_vocab tokens we stop merging.
    def train(self, dataset, max_vocab):
        chars = sorted(list(set(dataset)))
        assert len(chars) < max_vocab, "Dataset has more base tokens than `max_vocab`"

        # Vocabulary
        self.vocab = {char.encode('utf-8'): i for i, char in enumerate(chars)} # Build this?
        

        pass

    def encode(self, text):

        pass

    def decode(self, token):
        pass

    # Recieves the byte representation of the text, returns the most occurrent pair?
    # what if equality?    
    def _occur_freq(self):
        pass


with open('taylorswift.txt') as f:
    data = f.read()

chars = sorted(list(set(data)))
print(chars)
print(chars[0].encode('utf-8'))

utf_chars = [char.encode('utf-8') for char in chars]
# utf_chars = {char.encode('utf-8'): i for i, char in enumerate(chars)}
# print(utf_chars)
# print([word for word in list(data)[:20]])

print(len(utf_chars))
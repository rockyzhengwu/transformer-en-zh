# -*-coding:utf-8-*-

_PAD = "<pad>"
_UNK = "<unk>"
_EOS = '<eos>'
_PRE_TOKENS = [_PAD, _UNK, _EOS]
EOS_ID = 2
class TextEncoder(object):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.word_to_id = {}
        self.id_to_word = {}
        self._init()

    def _init(self):
        with open(self.vocab_file) as f:
            for _, line in enumerate(f):
                w = line.strip("\n").strip()
                if not w:
                    continue
                self.word_to_id[w] = len(self.word_to_id)


    def encode(self, text):
        words = text.split()
        ids = [self.word_to_id.get(w, self.word_to_id.get(_UNK)) for w in words]
        ids.append(self.word_to_id.get(_EOS))
        return ids


    def decode(self, ids):
        if(len(self.id_to_word) == 0):
            self.id_to_word = {v:k for k, v in self.word_to_id.items()}
        words = [self.id_to_word.get(index) for index in ids]
        return " ".join(words)


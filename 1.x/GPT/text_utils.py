import json
import spacy

def encode_dataset(*splits, encoder):
    encoded_splits = []
    for split in splits[0]:
        fields = []
        for field in split:
            if isinstance(field[0], str):
                field = encoder.encode(field)
            fields.append(field)
        encoded_splits.append(fields)
    return encoded_splits
    

class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load("en", disable=["parser", "tagger", "ner", "textcat"])
        self.encoder = json.load(open(ecoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path).read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

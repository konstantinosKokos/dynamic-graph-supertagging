# todo: assert homogeneous tokenization in encode sentence (inference) and encode words (training)

from .preprocessing2 import Sample, Symbol

from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch import Tensor
from transformers import BertTokenizer


def pad_sequence(sequence: list[Tensor], **kwargs) -> Tensor:
    return _pad_sequence(sequence, **kwargs)


class Tokenizer:
    def __init__(self):
        self.core = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')

    def encode_words(self, words: list[str]) -> tuple[list[int], list[int]]:
        subword_tokens = [[self.core.pad_token_id]] + \
                         [self.core.encode(w, add_special_tokens=False) for w in words] + \
                         [[self.core.sep_token_id]]
        word_clusters = [[i + 1] * len(subword_tokens[i]) for i in range(len(subword_tokens))]
        return sum(subword_tokens, []), sum(word_clusters, [])

    def encode_sentence(self, sentence: str) -> tuple[list[int], list[int]]:
        return self.encode_words(sentence.split())

    def encode_sample(self, sample: Sample) -> tuple[list[int], list[int]]:
        return self.encode_words(sample.words)

    def encode_samples(self, samples: list[Sample]) -> tuple[list[list[int]], list[list[int]]]:
        encoded = [self.encode_sample(s) for s in samples]
        return [e[0] for e in encoded], [e[1] for e in encoded]


class AtomTokenizer:
    def __init__(self, symbol_map: dict[int, Symbol], symbol_arities: dict[Symbol, int]):
        self.id_to_token = symbol_map
        self.symbol_arities = symbol_arities
        self.token_to_id = {v: k for k, v in symbol_map.items()}
        self.pad_token_id = self.token_to_id['[PAD]']
        self.sos_token_id = self.token_to_id['[SOS]']
        self.sep_token_id = self.token_to_id['[SEP]']

    def __len__(self) -> int: return len(self.id_to_token)
    def atom_to_id(self, atom: Symbol) -> int: return self.token_to_id[atom]
    def id_to_atom(self, idx: int) -> Symbol: return self.id_to_token[idx]

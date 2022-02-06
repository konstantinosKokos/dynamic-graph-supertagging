"""
    Contains utility functions to convert the aethel dataset to inputs for the neural proof net.
"""

from __future__ import annotations
import pdb
from dataclasses import dataclass

from LassyExtraction.extraction import Atoms
from LassyExtraction.aethel import Sample as AethelSample, aethel
from LassyExtraction.mill.nets import Tree as FTree, Unary as FUnary, Binary as FBinary, Leaf as FLeaf, term_to_links

from .tree import Tree, Unary, Binary, Leaf


@dataclass
class Symbol:
    __match_args__ = ('name',)

    name: str
    index: int | None = None
    logprob: float | None = None

    def __repr__(self) -> str: return self.name if self.index is None else f"{self.name}:{self.index}"
    def __eq__(self, other) -> bool: return isinstance(other, Symbol) and self.name == other.name
    def __hash__(self) -> int: return hash((self.name, self.index))

    def plain(self) -> Symbol:
        return Symbol(self.name, None, None)


MWU = Leaf(Symbol('MWU'))


def formula_tree_to_tree(formula_tree: FTree) -> Tree[Symbol]:
    match formula_tree:
        case FLeaf(atom, _, index):
            return Leaf(Symbol(atom, index))
        case FUnary(_, modality, decoration, content):
            return Unary(Symbol(modality + decoration), formula_tree_to_tree(content))
        case FBinary(_, left, right):
            return Binary(Symbol('->'), formula_tree_to_tree(left), formula_tree_to_tree(right))
        case _: raise ValueError(f'Unknown formula tree type: {formula_tree}')


def tree_to_formula_tree(tree: Tree[Symbol], polarity: bool = True) -> FTree:
    match tree:
        case Leaf(symbol):
            return FLeaf(symbol.name, polarity, symbol.index)
        case Unary(symbol, content):
            return FUnary(polarity, symbol.name[0], symbol.name[1:], tree_to_formula_tree(content, polarity))
        case Binary(_, left, right):
            return FBinary(polarity, tree_to_formula_tree(left, not polarity), tree_to_formula_tree(right, polarity))
        case _: raise ValueError(f'Unknown tree type: {tree}')


def binarize(tree: Tree[Symbol]) -> Tree[Symbol]:
    match tree:
        case Leaf(_):
            return tree
        case Unary(Symbol(outer), Binary(Symbol(inner), left, right)):
            return Binary(Symbol(outer + inner), binarize(left), binarize(right))
        case Binary(Symbol(outer), Unary(Symbol(inner), left), right):
            return Binary(Symbol(outer + inner), binarize(left), binarize(right))
        case Binary(symbol, left, right):
            return Binary(symbol, binarize(left), binarize(right))
        case _: raise ValueError(f'Unknown tree type: {tree}')


def debinarize(tree: Tree[Symbol]) -> Tree[Symbol]:
    match tree:
        case Leaf(_):
            return tree
        case Binary(Symbol(outer), left, right):
            if outer == '->':
                return Binary(Symbol(outer), debinarize(left), debinarize(right))
            if outer.startswith('->'):
                return Binary(Symbol('->'), Unary(Symbol(outer[2:]), debinarize(left)), debinarize(right))
            elif outer.endswith('->'):
                return Unary(Symbol(outer[:-2]), Binary(Symbol('->'), debinarize(left), debinarize(right)))
            else:
                raise ValueError(f'Unknown binary symbol: {outer}')
        case _: raise ValueError(f'Unknown tree type: {tree}')


def index_tree(tree: Tree[Symbol], index: int = 0) -> tuple[Tree[Symbol], int]:
    match tree:
        case Leaf(Symbol('MWU')):
            return tree, index
        case Leaf(Symbol(name, _)):
            return Leaf(Symbol(name, index)), index + 1
        case Unary(symbol, content):
            content, index = index_tree(content, index)
            return Unary(symbol, content), index
        case Binary(symbol, left, right):
            left, index = index_tree(left, index)
            right, index = index_tree(right, index)
            return Binary(symbol, left, right), index
        case _: raise ValueError(f'Unknown tree type: {tree}')


def pad_mwus(words: list[str], types: list[Tree]) -> tuple[list[str], list[Tree]]:
    words = [w.split() for w in words]
    types = [[t] + [MWU] * (len(ws) - 1) for ws, t in zip(words, types)]
    return [w for units in words for w in units], [t for units in types for t in units]


@dataclass
class Sample:
    # todo: distinguish between inference and training samples
    # todo: index-generic atomsets and matrices
    words: list[str]
    trees: list[Tree[Symbol]]
    links: dict[Tree[Symbol], Tree[Symbol]]
    source: str | None = None

    @staticmethod
    def from_aethel(sample: AethelSample) -> 'Sample':
        premises = sample.premises
        links, participating_trees = term_to_links(sample.proof)

        trees = [formula_tree_to_tree(participating_trees[i])
                 if i in participating_trees.keys()
                 else Leaf(Symbol(premises[i].type.sign))  # type: ignore
                 for i in range(len(sample.premises))]
        words = [p.word for p in premises]
        words, trees = pad_mwus(words, trees)
        return Sample(
            words=words,
            trees=trees,
            links={formula_tree_to_tree(neg): formula_tree_to_tree(pos) for neg, pos in links.items()},
            source=sample.name)

    def to_aethel(self):
        # todo: mwu-unification, punct ignoring
        links = {tree_to_formula_tree(neg, False): tree_to_formula_tree(pos) for neg, pos in self.links.items()}
        trees = [tree_to_formula_tree(t) for t in self.trees]
        return links, trees


def make_symbol_map(symbols: set[tuple[Symbol, int | None]]) -> tuple[dict[int, Symbol], dict[Symbol, int | None]]:
    special = [(Symbol('[PAD]'), None), (Symbol('[SOS]'), None), (Symbol('[EOS]'), None)]
    sorted_symbols = special + sorted(symbols, key=lambda s: s[0])
    id_to_symbol = {i: s for i, s in enumerate([s for s, _ in sorted_symbols])}
    symbol_to_arity = {s: a for s, a in sorted_symbols}
    return id_to_symbol, symbol_to_arity


def main(load_path: str = '../lassy-tlg-extraction/data/aethel.pickle'):
    data = [Sample.from_aethel(d) for d in aethel.load_data(load_path)]
    symbols = set([(symbol.plain(), arity)
                   for sample in data
                   for tree in sample.trees
                   for symbol, arity in tree.nodes_and_arities()])
    id_to_symbol, symbol_to_arity = make_symbol_map(symbols)


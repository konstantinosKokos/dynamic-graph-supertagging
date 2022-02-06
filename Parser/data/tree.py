from abc import ABC
from typing import TypeVar, Generic, Any, Callable
from itertools import zip_longest


Node = TypeVar('Node')
Other = TypeVar('Other')


class Tree(ABC, Generic[Node]):
    node: Node

    def depth(self) -> int: return tree_depth(self)
    def __eq__(self, other: Any) -> bool: return tree_eq(self, other)
    def leaves(self) -> list[Node]: return tree_leaves(self)
    def __hash__(self) -> int: return tree_hash(self)
    def __repr__(self) -> str: return tree_repr(self)
    def nodes_and_arities(self) -> list[tuple[Node, int]]: return nodes_and_arities(self)
    def fmap(self, f: Callable[[Node], Other]) -> 'Tree[Other]': return tree_fmap(self, f)


class Leaf(Tree[Node]):
    __match_args__ = ('node',)

    def __init__(self, node: Node):
        self.node = node


class Unary(Tree[Node]):
    __match_args__ = ('node', 'child')

    def __init__(self, node: Node, child: Tree[Node]):
        self.node = node
        self.child = child


class Binary(Tree[Node]):
    __match_args__ = ('node', 'left', 'right')

    def __init__(self, node: Node, left: Tree[Node], right: Tree[Node]):
        self.node = node
        self.left = left
        self.right = right


def tree_depth(tree: Tree[Node]) -> int:
    match tree:
        case Leaf(_):
            return 0
        case Unary(_, child):
            return 1 + tree_depth(child)
        case Binary(_, left, right):
            return 1 + max(tree_depth(left), tree_depth(right))


def tree_eq(left: Tree[Node], right: Tree[Node]) -> bool:
    match left, right:
        case Leaf(lnode), Leaf(rnode):
            return lnode == rnode
        case Unary(lnode, lchild), Unary(rnode, rchild):
            return lnode == rnode and tree_eq(lchild, rchild)
        case Binary(lnode, lleft, lright), Binary(rnode, rleft, rright):
            return lnode == rnode and tree_eq(lleft, rleft) and tree_eq(lright, rright)
        case _:
            return False


def tree_leaves(tree: Tree[Node]) -> list[Node]:
    return list(depth_first(tree))


def tree_hash(tree: Tree[Node]) -> int:
    match tree:
        case Leaf(node): return hash((node,))
        case Unary(node, child): return hash((node, child,))
        case Binary(node, left, right): return hash((node, left, right))
        case _: raise TypeError(f'{tree} is not a tree')


def tree_repr(tree: Tree[Node]) -> str:
    match tree:
        case Leaf(node): return f'Leaf({node})'
        case Unary(node, child): return f'Unary({node}, {tree_repr(child)})'
        case Binary(node, left, right): return f'Binary({node}, {tree_repr(left)}, {tree_repr(right)})'
        case _: raise TypeError(f'{tree} is not a tree')


def nodes_and_arities(tree: Tree[Node]) -> list[tuple[Node, int]]:
    match tree:
        case Leaf(node): return [(node, 0)]
        case Unary(node, child): return [(node, 1)] + nodes_and_arities(child)
        case Binary(node, left, right): return [(node, 2)] + nodes_and_arities(left) + nodes_and_arities(right)
        case _: raise TypeError(f'{tree} is not a tree')


def tree_fmap(tree: Tree[Node], f: Callable[[Node], Other]) -> 'Tree[Other]':
    match tree:
        case Leaf(node): return Leaf(f(node))
        case Unary(node, child): return Unary(f(node), tree_fmap(child, f))
        case Binary(node, left, right): return Binary(f(node), tree_fmap(left, f), tree_fmap(right, f))
        case _: raise TypeError(f'{tree} is not a tree')


def depth_first(tree: Tree[Node]) -> list[Node]:
    match tree:
        case Leaf(node): return [node]
        case Unary(node, child): return [node] + depth_first(child)
        case Binary(node, left, right): return [node] + depth_first(left) + depth_first(right)
        case _: raise TypeError(f'{tree} is not a tree')


def levels(tree: Tree[Node]) -> list[list[Node]]:
    match tree:
        case Leaf(node): return [[node]]
        case Unary(node, child): return [[node]] + levels(child)
        case Binary(node, left, right):
            return [[node]] + [sum(xs, []) for xs in zip_longest(levels(left), levels(right), fillvalue=[])]


def breadth_first(tree: Tree[Node]) -> list[Node]:
    return sum(levels(tree), [])


def dfs_to_tree(sequence: list[Node]) -> Tree[Node]:
    ...


def bfs_to_tree(sequence: list[Node]) -> Tree[Node]:
    ...



# tree = Binary(0, Binary(1, Leaf(3), Binary(4, Leaf(6), Unary(7, Leaf(9)))), Unary(2, Unary(5, Leaf(8))))
#
# # print(list(depth_first(tree)))
# print(list(levels(tree)))

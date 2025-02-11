#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from copy import deepcopy

from .node import Node, NodeType

from .utils import arity
from .my_random import gxgp_random
import operator
import numpy as np


def xover_swap_subtree(tree1: Node, tree2: Node) -> Node:
    offspring = deepcopy(tree1)
    # Force evaluation
    subtree_list = list(offspring.subtree)
    if not subtree_list:
        raise ValueError("Tree1 has no valid subtree for crossover.")

    successors = None
    while not successors:
        node = gxgp_random.choice(subtree_list)
        successors = node.successors
    i = gxgp_random.randrange(len(successors))
    replacement_subtree = list(tree2.subtree)
    if not replacement_subtree:
        raise ValueError("Tree2 has no valid subtree for crossover.")
    
    successors[i] = deepcopy(gxgp_random.choice(replacement_subtree))
    node.successors = successors
    return offspring

def mutate_point(tree: Node, num_variables, operators) -> Node:
    node_to_mutate = tree
    if node_to_mutate.type == NodeType.FUNCTION:
        while True:
            op = gxgp_random.choice(operators)
            if node_to_mutate.arity != arity(op):
                continue
            n= Node(op, node_to_mutate.successors)
            if n.short_name != node_to_mutate.short_name:
                break
        
    elif node_to_mutate.type == NodeType.CONSTANT:
        if gxgp_random.random() < 0.8:
            n= Node(gxgp_random.uniform(-2, 2))
        else:
            j = gxgp_random.randint(0, num_variables - 1) if num_variables > 1 else 0
            n = Node(f'x{j}')
    elif node_to_mutate.type == NodeType.VARIABLE:
        if gxgp_random.random() < 0.8 and num_variables > 1:
            i = int(node_to_mutate.short_name[1:])
            while True:
                j = gxgp_random.randint(0, num_variables - 1)
                if j != i:
                    break
            n = Node(f'x{j}')
        else:
            n = Node(gxgp_random.uniform(-2, 2))

    else:
        assert False, f'Unknown node type: {type(node_to_mutate)}'

    tree = n

    return tree


def mutate_random_points(tree: Node, num_variables: int, operators, probability: float = 0.5) -> Node:
    mask = np.random.choice([True, False], len(tree), p=[probability, 1-probability])
    while not np.any(mask):
        mask = np.random.choice([True, False], len(tree), p=[probability, 1-probability])

    
    def traverse_and_mutate(node, mask, idx=0):
        if idx >= len(mask):
            return node, idx

        if mask[idx]:
            node = mutate_point(node, num_variables, operators)

        
        succ = node.successors
        for i in range(len(node.successors)):
            succ[i], idx = traverse_and_mutate(node.successors[i], mask, idx + 1)

        node.successors = succ
        
        return node, idx
    
    mutant = deepcopy(tree)

    mutated_tree, _ = traverse_and_mutate(mutant, mask)
    return mutated_tree

def mutate_subtree(tree: Node, num_variables: int, operators, idx=None) -> Node:
    mutant = deepcopy(tree)
    
    if idx is None:
        idx = gxgp_random.randint(0, len(mutant) - 1)

    def traverse_and_mutate(node, idx, current_idx=0):
        if current_idx == idx:
            if node.is_leaf:
                op = gxgp_random.choice(operators)
                successors = []
                for _ in range(arity(op)):
                    if gxgp_random.random() < 0.5:
                        successors.append(Node(gxgp_random.uniform(-2, 2)))
                    else:
                        j = gxgp_random.randint(0, num_variables - 1)
                        successors.append(Node(f'x{j}'))
                return Node(op, successors), current_idx + 1
            else:
                op = gxgp_random.choice([operator.add, operator.sub, operator.mul])
                successors = []
                if gxgp_random.random() < 0.5:
                        successors.append(Node(gxgp_random.uniform(-2, 2)))
                else:
                    j = gxgp_random.randint(0, num_variables - 1)
                    successors.append(Node(f'x{j}'))
                successors.append(node)
                return Node(op, successors), current_idx + 1
        
        succ = node.successors
        for i in range(len(node.successors)):
            succ[i], current_idx = traverse_and_mutate(node.successors[i], idx, current_idx + 1)

        node.successors = succ

        return node, current_idx
    
    mutated_tree, _ = traverse_and_mutate(mutant, idx)
    return mutated_tree
        


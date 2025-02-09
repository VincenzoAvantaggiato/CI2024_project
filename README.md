# Project Work: Symbolic Regression

## Introduction
The goal of this project is to perform symbolic regression using an evolutionary approach. Specifically, I implemented `swap subtree crossover` and a combination of `point mutation` and `subtree mutation` to evolve mathematical expressions that fit given datasets.

For each problem, I tuned a series of hyperparameters, which are summarized in the following table.

## Hyperparameter Tuning
### Table 1: Hyperparameters

| Problem | Population Size | Generations | Offspring Size | Individual Size | Tournament Size | Resize Factor | Mutation Const | Mutation Rate | Length Penalty | Crossover Rate |
|---------|-----------------|-------------|----------------|-----------------|-----------------|---------------|----------------|---------------|----------------|----------------|
| 0       | 100             | 100         | 30             | 6               | 5               | 500           | 0.25           | 0.25          | 0              | 0.5            |
| 1       | 100             | 50          | 30             | 6               | 5               | 500           | 0.25           | 0             | 0              | 0.5            |
| 2       | 100             | 400         | 30             | 6               | 5               | 500           | 0.25            | 0.5          | 0              | 0.5            |
| 3       | 100             | 400         | 30             | 6               | 5               | 500           | 0.5            | 0             | 1e-2           | 0.5            |
| 4       | 100             | 300         | 30             | 6               | 5               | 200           | 0.25           | 0             | 0              | 0.5            |
| 5       | 100             | 500         | 30             | 6               | 5               | 500           | 0.5            | 0             | 1e-2           | 0.5            |
| 6       | 100             | 400         | 30             | 6               | 5               | 500           | 0.25           | 0             | 0              | 0.5            |
| 7       | 100             | 300         | 30             | 6               | 5               | 5000          | 0.4            | 0.2           | 1e-2           | 0.5            |
| 8       | 100             | 500         | 30             | 6               | 5               | 5000          | 0.5            | 0             | 1e-2           | 0.5            |

### Key Hyperparameters
- **Resize Factor**: Determines the number of samples taken from the original dataset. The hypothesis (which was confirmed) was that a large number of samples was unnecessary, so I downsampled by selecting every `n`th sample to speed up computation.
- **Length Penalty**: Penalizes longer solutions to control complexity. The computational bottleneck is the recursive function evaluation, which is linear in its size.
- **Mutation Constants and Rates**: These determine the probability of selecting one type of mutation over another, calculated as `MUT_A + MUT_B * (generation / GENERATIONS)`.

## Implementation Changes
I based my implementation on the `gxgp` library developed by Professor Squillero, making several optimizations and additions. Minor changes aimed to enhance performance, while major additions introduced new functionalities.

### Notable Additions
One of the significant additions is the implementation of enhanced mutation functions:

``` python
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
```

More details and explanations are provided in the notebook.

## Results
### Table 2: Formula, MSE, and Length

| Problem | Formula | MSE | Length |
|---------|---------|-----|--------|
| 0       | ![](images/problem0/formula.png) | ![](images/problem0/mse.png) | ![](images/problem0/length.png) |
| 1       | ![](images/problem1/formula.png) | ![](images/problem1/mse.png) | ![](images/problem1/length.png) |
| 2       | ![](images/problem2/formula.png) | ![](images/problem2/mse.png) | ![](images/problem2/length.png) |
| 3       | ![](images/problem3/formula.png) | ![](images/problem3/mse.png) | ![](images/problem3/length.png) |
| 4       | ![](images/problem4/formula.png) | ![](images/problem4/mse.png) | ![](images/problem4/length.png) |
| 5       | ![](images/problem5/formula.png) | ![](images/problem5/mse.png) | ![](images/problem5/length.png) |
| 6       | ![](images/problem6/formula.png) | ![](images/problem6/mse.png) | ![](images/problem6/length.png) |
| 7       | ![](images/problem7/formula.png) | ![](images/problem7/mse.png) | ![](images/problem7/length.png) |
| 8       | ![](images/problem8/formula.png) | ![](images/problem8/mse.png) | ![](images/problem8/length.png) |

### Table 3: Plots
Each plot was generated by setting all variables to zero except for the one being plotted.

| Problem | Plots per component | 3D Plot |
|---------|---------------------|---------|
| 0       | <img src="images/problem0/x0.png"/><br><img src="images/problem0/x1.png"/> | <img src="images/problem0/3d.png"/> |
| 1       | <img src="images/problem1/x0.png"/> |  |
| 2       | <img src="images/problem2/x0.png"/><br><img src="images/problem2/x1.png"/><br><img src="images/problem2/x2.png"/> |  |
| 3       | <img src="images/problem3/x0.png"/><br><img src="images/problem3/x1.png"/><br><img src="images/problem3/x2.png"/> |  |
| 4       | <img src="images/problem4/x0.png"/><br><img src="images/problem4/x1.png"/> | <img src="images/problem4/3d.png"/> |
| 5       | <img src="images/problem5/x0.png"/><br><img src="images/problem5/x1.png"/> | <img src="images/problem5/3d.png"/> |
| 6       | <img src="images/problem6/x0.png"/><br><img src="images/problem6/x1.png"/> | <img src="images/problem6/3d.png"/> |
| 7       | <img src="images/problem7/x0.png"/><br><img src="images/problem7/x1.png"/> | <img src="images/problem7/3d.png"/> |
| 8       | <img src="images/problem8/x0.png"/><br><img src="images/problem8/x1.png"/><br><img src="images/problem8/x2.png"/><br><img src="images/problem8/x3.png"/><br><img src="images/problem8/x4.png"/><br><img src="images/problem8/x5.png"/> |  |

## Conclusion
This project successfully implemented symbolic regression using evolutionary strategies. The performance optimizations and added functionalities contributed to faster execution and improved model interpretability. Future work could explore alternative crossover methods and more sophisticated regularization techniques to further refine symbolic regression solutions.





### Table 4: Results

| Problem | Formula | MSE * 100 |
|---------|---------|-----|
| 0       | `np.add(x[0], np.multiply(-0.140281, np.multiply(np.subtract(-0.400897, np.cos(np.multiply(np.sin(np.cos(x[1])), np.subtract(np.multiply(x[1], x[1]), np.multiply(np.multiply(np.sin(x[1]), 0.426575), np.multiply(np.subtract(-0.140281, np.cos(np.multiply(np.sin(x[1]), np.subtract(1.25265, np.multiply(np.multiply(np.sin(x[1]), 0.426575), np.multiply(np.subtract(-0.140281, np.multiply(np.multiply(np.sin(x[1]), 0.426575), np.multiply(np.subtract(-0.140281, np.sin(np.subtract(np.subtract(0.916315, np.sin(np.sin(np.multiply(1.28782, x[1])))), np.multiply(-0.140281, x[1])))), np.multiply(1.28782, x[1])))), np.multiply(1.28782, x[1]))))))), x[1])))))), x[1])))`       | `9.27464e-05`   |
| 1       | `np.sin(x[0])`       | `7.12594e-32`   |
| 2       | `np.add(np.multiply(np.add(np.multiply(np.multiply(x[0], np.sin(-0.898513)), 0.0545983), np.add(x[0], np.add(1.28009, np.sinh(np.multiply(np.exp(1.68327), np.add(0.987022, np.multiply(-1.69614, np.tanh(-1.56689)))))))), x[2]), np.multiply(np.add(x[1], np.add(np.add(0.987022, np.multiply(-1.69614, np.tanh(x[0]))), np.subtract(np.sinh(np.multiply(np.exp(1.68327), 1.7515)), np.add(np.multiply(1.70039, np.add(0.36735, np.cosh(np.add(x[2], x[0])))), np.cosh(np.add(np.multiply(0.328658, x[0]), np.add(x[2], x[1]))))))), np.add(np.multiply(np.exp(x[1]), 1.7515), np.add(x[0], np.add(np.exp(x[1]), np.multiply(x[0], np.exp(np.exp(1.68327))))))))`       | `1.44782e+15`   |
| 3       | `np.add(np.subtract(np.multiply(np.sin(0.516614), np.subtract(np.subtract(x[1], x[1]), x[2])), x[2]), np.add(np.subtract(np.sin(np.multiply(1.0099, 0.799579)), x[2]), np.add(np.subtract(np.sin(0.729197), x[2]), np.add(1.38768, np.add(np.multiply(x[0], x[0]), np.add(np.multiply(x[0], x[0]), np.add(np.multiply(np.multiply(np.multiply(x[1], np.cos(-1.23315)), np.sin(np.multiply(1.11106, np.multiply(1.0099, 0.729197)))), 0.799579), np.add(np.subtract(1.19791, x[1]), np.multiply(np.subtract(0.799579, np.multiply(x[1], x[1])), x[1])))))))))`       | `0.134277`   |
| 4       | `np.add(np.cos(np.multiply(1.37954, -0.295438)), np.add(-1.44507, np.add(np.add(np.cos(x[1]), np.add(1.37954, np.add(np.add(np.cos(np.cos(np.subtract(np.add(np.multiply(0.0632359, np.multiply(-0.891081, np.multiply(-1.4886, np.multiply(-0.946446, np.multiply(-1.4886, np.cos(np.multiply(np.subtract(0.859964, x[0]), np.cos(np.subtract(-0.946446, np.cos(np.subtract(0.591412, np.cos(np.add(np.multiply(0.0632359, np.multiply(0.0569691, np.multiply(-1.4886, np.cos(np.multiply(x[1], np.cos(np.cos(np.subtract(np.multiply(0.0569691, np.multiply(np.subtract(-0.946446, np.cos(np.subtract(0.591412, np.cos(np.cos(np.subtract(0.591412, np.cos(np.multiply(1.42285, np.add(np.multiply(np.cos(np.subtract(-0.946446, np.cos(-0.295438))), np.subtract(np.cos(x[1]), x[1])), np.add(np.add(np.cos(x[1]), np.add(1.37954, np.add(np.add(np.cos(np.cos(np.subtract(-0.946446, np.cos(np.add(np.multiply(0.0632359, np.multiply(-0.891081, np.multiply(-1.4886, np.cos(np.cos(np.subtract(np.cos(np.multiply(np.subtract(0.859964, x[0]), np.cos(x[1]))), np.cos(-0.295438))))))), -0.891081))))), np.add(np.cos(x[1]), np.cos(x[1]))), np.add(np.cos(0.591412), np.add(0.859964, np.add(-0.00216417, np.cos(x[1]))))))), np.add(np.cos(x[1]), np.add(np.cos(np.subtract(np.multiply(0.0569691, np.multiply(np.subtract(-0.946446, np.cos(np.subtract(np.cos(np.cos(np.subtract(np.multiply(0.0569691, np.multiply(np.subtract(-0.946446, np.cos(np.subtract(0.591412, np.cos(np.cos(np.subtract(0.591412, np.multiply(np.subtract(0.859964, x[0]), np.cos(x[1])))))))), x[0])), np.cos(np.subtract(np.subtract(x[0], x[0]), -0.295438))))), np.cos(np.cos(np.subtract(0.591412, np.cos(np.multiply(np.cos(np.subtract(-0.946446, np.cos(np.add(np.multiply(0.0632359, np.multiply(-0.891081, np.multiply(np.cos(np.multiply(x[1], np.cos(x[1]))), np.cos(np.add(-1.25967, np.subtract(x[0], 1.60493)))))), -0.891081)))), np.add(np.multiply(np.cos(np.subtract(-0.946446, np.cos(-0.295438))), np.subtract(-0.386744, x[1])), 0.0569691))))))))), x[0])), np.multiply(-1.50655, np.cos(np.subtract(np.subtract(np.sin(x[1]), np.add(0.678777, x[0])), -0.295438))))), np.cos(x[1]))))))))))))), x[0])), np.cos(np.subtract(np.subtract(x[0], x[0]), -0.295438)))))))))), -0.891081))))))))))))), -0.891081), np.cos(np.add(np.multiply(0.0632359, np.multiply(0.0569691, np.multiply(-1.4886, np.cos(np.subtract(np.multiply(0.0569691, np.multiply(np.multiply(0.545816, np.subtract(-0.946446, np.cos(np.subtract(0.591412, np.cos(np.cos(np.subtract(0.591412, np.cos(np.multiply(1.42285, np.add(np.multiply(x[0], np.subtract(np.cos(x[1]), x[1])), np.add(np.add(np.cos(x[1]), np.add(1.37954, np.add(np.add(np.cos(np.cos(np.subtract(-0.946446, 0.859964))), np.add(np.cos(x[1]), np.cos(x[1]))), np.add(np.cos(x[1]), np.add(0.859964, np.add(-0.00216417, np.cos(x[1]))))))), np.add(np.cos(x[1]), np.add(np.cos(np.subtract(np.multiply(0.0569691, np.multiply(np.subtract(-0.946446, np.cos(np.subtract(0.591412, np.cos(np.cos(np.subtract(0.591412, np.cos(np.multiply(np.cos(np.subtract(-0.946446, np.cos(np.add(np.multiply(0.0632359, np.multiply(-0.891081, np.subtract(x[1], np.multiply(np.cos(np.multiply(x[1], np.cos(x[1]))), np.cos(np.add(-1.25967, np.subtract(np.cos(x[1]), 1.60493))))))), -0.891081)))), np.add(np.multiply(np.cos(np.subtract(np.cos(np.multiply(np.subtract(0.859964, x[0]), np.cos(x[1]))), np.cos(-0.295438))), np.subtract(-0.386744, x[1])), 0.0569691))))))))), x[0])), np.cos(np.subtract(np.subtract(np.sin(x[1]), np.add(0.678777, x[0])), -0.295438)))), np.cos(x[1])))))))))))))), x[0])), np.multiply(x[0], np.cos(np.subtract(np.subtract(-0.386744, x[0]), -0.295438)))))))), -0.891081))))), np.add(np.cos(x[1]), np.cos(x[1]))), np.add(np.cos(x[1]), np.add(0.859964, np.add(-0.00216417, np.cos(x[1]))))))), np.add(np.cos(x[1]), np.add(np.cos(np.subtract(np.multiply(0.0569691, np.multiply(np.subtract(-0.946446, np.cos(np.subtract(0.591412, np.cos(np.add(np.multiply(0.0632359, np.multiply(0.0569691, np.multiply(-1.4886, np.cos(np.cos(x[1]))))), -0.891081))))), x[0])), np.cos(np.subtract(np.subtract(x[0], np.add(0.678777, x[0])), -0.295438)))), np.cos(x[1]))))))`       | `0.0721357`   |
| 5       | `np.multiply(x[0], np.multiply(0.363602, np.multiply(np.add(np.subtract(x[0], 0.363602), np.cos(x[0])), np.multiply(np.cos(-1.50301), np.multiply(0.363602, np.multiply(np.multiply(np.multiply(np.multiply(np.subtract(x[1], 0.698396), np.multiply(np.cos(-1.50301), np.exp(-1.50301))), np.multiply(0.363602, np.multiply(np.add(x[0], np.subtract(x[1], 0.698396)), np.multiply(np.multiply(np.cos(-1.50301), np.multiply(np.subtract(x[1], np.subtract(np.cos(np.exp(np.subtract(-1.52497, x[1]))), -0.256538)), np.multiply(np.cos(-1.50301), np.exp(-1.50301)))), np.multiply(np.multiply(np.subtract(x[1], np.cos(np.add(x[0], x[1]))), np.multiply(np.cos(-1.50301), np.exp(-1.80762))), np.exp(-1.77607)))))), np.cos(-1.77607)), np.multiply(np.multiply(np.subtract(np.cos(np.cos(np.subtract(x[1], 0.698396))), np.cos(-1.77607)), np.exp(-1.50301)), np.exp(-1.50301))))))))`       | `1.13839e-17`   |
| 6       | `np.add(np.multiply(np.sin(0.784181), x[1]), np.subtract(np.add(np.multiply(1.34051, np.multiply(x[1], np.multiply(np.cos(x[1]), np.multiply(x[0], np.subtract(x[0], x[0]))))), x[1]), np.add(np.multiply(np.subtract(np.add(0.784181, -0.0506562), 0.784181), np.subtract(0.560634, np.multiply(0.0874461, np.subtract(np.add(1.94535, np.add(x[1], np.cos(np.add(np.multiply(np.subtract(np.add(np.add(x[1], np.add(np.subtract(x[1], np.multiply(x[1], np.add(x[1], np.multiply(np.subtract(np.cos(x[1]), np.multiply(np.subtract(0.560634, np.multiply(0.0874461, np.add(x[1], np.subtract(np.add(1.94535, np.add(x[1], np.cos(np.add(np.multiply(0.784181, -0.0506562), np.multiply(x[0], np.subtract(0.587439, x[0])))))), np.add(np.subtract(-1.86602, x[0]), np.subtract(np.multiply(x[0], 1.79658), x[1])))))), np.sin(0.784181))), np.add(np.multiply(np.subtract(np.add(0.784181, -0.0506562), 0.784181), np.subtract(np.add(np.multiply(1.34051, np.multiply(x[1], np.multiply(np.cos(x[1]), np.subtract(0.560634, np.multiply(0.0874461, np.subtract(np.add(1.94535, np.add(x[1], np.cos(np.add(np.multiply(np.subtract(np.add(np.add(np.subtract(x[1], np.multiply(x[1], np.add(x[1], np.multiply(np.subtract(np.cos(np.multiply(0.109979, -1.22475)), np.multiply(np.subtract(x[1], np.multiply(0.0874461, np.add(x[1], np.subtract(np.add(1.94535, np.add(np.multiply(np.subtract(x[1], np.cos(np.subtract(0.587439, x[0]))), np.multiply(x[0], np.subtract(np.add(np.multiply(np.subtract(np.add(0.784181, -0.0506562), -0.0506562), x[1]), np.multiply(x[0], np.sin(0.784181))), x[0]))), np.subtract(0.539703, np.cos(np.add(np.multiply(0.784181, -0.0506562), np.add(np.multiply(np.subtract(0.438263, x[0]), np.multiply(np.multiply(x[0], -1.135), np.multiply(np.cos(x[1]), np.multiply(np.multiply(x[0], np.subtract(x[0], x[0])), np.subtract(x[0], x[0]))))), np.cos(np.multiply(x[1], np.multiply(np.cos(x[1]), np.multiply(x[0], np.subtract(x[0], x[0]))))))))))), np.add(np.subtract(-1.86602, -0.593448), np.subtract(np.multiply(x[0], np.add(1.45885, np.multiply(x[1], np.multiply(np.cos(x[1]), np.multiply(np.sin(-1.07768), np.subtract(x[0], x[0])))))), np.cos(x[0]))))))), np.sin(0.784181))), np.add(x[1], np.subtract(x[0], 0.784181)))))), x[1]), np.subtract(np.add(x[1], x[0]), -0.696719)), 0.784181), -0.0506562), np.multiply(np.multiply(0.0874461, np.add(0.471485, x[0])), -0.0506562))))), np.add(-0.83792, np.add(-1.02933, np.subtract(np.multiply(x[0], np.add(1.45885, np.cos(np.multiply(x[1], np.multiply(np.multiply(1.79658, np.subtract(-0.811335, np.cos(x[1]))), np.multiply(x[0], np.subtract(x[0], x[0]))))))), x[1]))))))))), x[1]), np.multiply(0.0874461, np.subtract(np.multiply(x[1], np.add(1.94535, np.add(1.4523, np.add(x[1], np.cos(np.add(np.multiply(np.subtract(np.add(np.add(np.subtract(x[1], np.multiply(x[1], np.add(x[1], np.multiply(np.subtract(np.cos(x[1]), np.multiply(np.add(np.multiply(np.subtract(np.add(-1.71706, np.add(0.784181, -0.0506562)), -0.0506562), x[1]), np.multiply(x[0], np.sin(0.784181))), np.sin(np.multiply(1.79522, 0.134326)))), np.add(np.sin(x[1]), np.subtract(x[0], 0.784181)))))), x[1]), np.subtract(np.add(x[1], x[0]), -0.696719)), 0.784181), -0.0506562), np.multiply(np.multiply(-1.37784, np.add(x[1], np.multiply(np.subtract(np.cos(x[1]), np.multiply(np.add(np.multiply(np.subtract(np.add(0.784181, -0.0506562), 0.784181), np.subtract(0.560634, np.multiply(0.0874461, np.subtract(np.add(1.94535, np.add(x[1], np.cos(np.add(np.multiply(np.subtract(np.add(np.add(np.subtract(x[1], np.multiply(x[1], np.add(np.sin(-0.609639), np.multiply(np.subtract(np.cos(x[1]), x[0]), np.add(x[1], np.subtract(x[0], 0.784181)))))), x[1]), np.subtract(np.add(x[1], x[0]), -0.696719)), 0.784181), -0.0506562), np.multiply(np.multiply(0.0874461, np.add(0.471485, x[0])), -0.0506562))))), np.add(-0.83792, np.add(-1.02933, np.subtract(np.multiply(x[0], np.add(1.45885, np.cos(np.multiply(x[1], np.multiply(np.multiply(1.79658, np.subtract(-0.811335, np.cos(x[1]))), np.multiply(x[0], np.subtract(x[0], x[0]))))))), x[1]))))))), np.multiply(x[0], np.sin(0.784181))), np.sin(0.784181))), np.add(0.341927, np.subtract(x[0], 0.784181))))), -0.0506562))))))), np.add(-0.83792, np.add(-1.02933, np.multiply(np.add(-0.630949, -0.311697), np.subtract(np.multiply(x[0], np.add(1.45885, np.cos(np.multiply(x[1], np.multiply(np.add(np.subtract(-1.86602, -0.593448), np.subtract(np.multiply(x[0], np.add(1.45885, np.multiply(x[1], np.add(1.83933, np.multiply(np.cos(x[1]), np.multiply(x[0], np.subtract(x[0], x[0]))))))), np.cos(x[0]))), np.multiply(x[0], np.subtract(x[0], x[0]))))))), np.sin(1.06108))))))))), np.multiply(x[0], np.sin(0.784181))))))), x[1])), np.subtract(x[0], -0.696719)), np.subtract(x[1], x[1])), -0.0506562), np.multiply(np.multiply(-1.37784, np.add(np.multiply(1.94535, np.sin(x[1])), x[0])), -0.0506562))))), np.add(-0.83792, np.add(-1.02933, np.subtract(np.multiply(x[0], np.add(1.45885, np.cos(np.multiply(x[1], np.multiply(np.cos(x[1]), np.multiply(x[0], np.subtract(x[0], x[0]))))))), x[1]))))))), np.multiply(x[0], np.sin(0.784181)))))`       | `0.00272643`   |
| 7       | `np.cosh(np.subtract(-0.169562, np.subtract(-0.381632, np.add(np.multiply(x[0], x[1]), np.exp(np.cos(np.multiply(np.add(np.multiply(x[0], x[1]), np.exp(np.cos(np.cosh(0.00138594)))), np.subtract(x[0], x[1]))))))))`       | `11613.6`   |
| 8       | `np.multiply(np.add(-1.73296, np.add(-0.991745, np.sinh(x[5]))), np.add(np.cosh(np.exp(1.68681)), np.exp(np.exp(1.48789))))`       | `7.38074e+07`   |
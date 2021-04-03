# Randomized MinCut Algorithm (FastCut)

This repository contains a Rust implementation of the "FastCut" randomized minimum cut algorithm
introduced by Karger and Stein (see Figure 6 in [A New Approach to the Minimum Cut Problem](https://doi.org/10.1145/234533.234534)).

I wrote the program for an oral examination for a randomized algorithms class at university.
It is probably not suitable for application to real-world problems.

![Example Of a Computed MinCut Visualized Using VANTED](https://raw.githubusercontent.com/5hir0kur0/RandomizedMinCut/assets/graph_200_split.png)

(A GML file output by this program. Visualized using [VANTED](https://www.cls.uni-konstanz.de/software/vanted/)).

## Features

- MinCut Estimation
- Parallelized Computation
- GML Output (No Layout)

## Usage

```sh
./mincut [-v] <PATH_TO_GRAPH>
```

If the `-v` is specified, a visualization of the result is written to `<PATH_TO_GRAPH>.gml`.

## Graph Format

This program processes undirected graphs with unit weights.
It reads plain text files that specify the edges.
The nodes are assumed to be numbered with integers from `0` to `<num_nodes>`.
The input format is as follows:
```text
<num_nodes>
<num_edges>
<source1> <target1>
<source2> <target2>
...
```

## Graph Representation

The graph is internally represented as an adjacency matrix.
Since it is undirected it is enough to store the upper triangular matrix.

Example matrix:

![Graph Data Structure Visualization](https://raw.githubusercontent.com/5hir0kur0/RandomizedMinCut/assets/data_structure_explanation.png)
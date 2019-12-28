use crate::matrix::Matrix;
use rand::{self, Rng};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

type EdgeCount = u32;

/// `T` is the type used to store the number of edges between nodes.
/// Nodes are always represented as indexes into a matrix of `T`,
/// so nodes don't have names and are just `usize` values.
#[derive(Debug)]
pub struct MultiGraph {
    edges: Matrix<EdgeCount>,
    cum_row_sums: Vec<usize>,
    rng: rand::rngs::ThreadRng,
    /// Stores which nodes are "contained" in which row/col in the matrix.
    /// After contracting an edge the two nodes at the beginning and at the end
    /// are represented by just one row/col in the matrix.
    node_to_row: Vec<usize>,
}

/// Try to parse the input as `T`. If the parsing fails
/// return an error with the error message "Expected {unit}".
/// The string is trimmed before parsing.
fn parse<T: std::str::FromStr>(input: &str, unit: &str) -> io::Result<T> {
    Ok(input.trim().parse::<T>().or_else(|_| {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Expected {}.", unit),
        ))
    })?)
}

/// Try to parse the input as `T`. If the input is not present or
/// the parsing fails return an error with the error message "Expected {unit}".
/// Calls `parse()` to do the parsing (see above).
fn read<T: std::str::FromStr>(input: Option<io::Result<String>>, unit: &str) -> io::Result<T> {
    let num = input.ok_or_else(|| {
        io::Error::new(io::ErrorKind::UnexpectedEof, format!("Expected {}.", unit))
    })??;
    parse(&num, unit)
}

impl MultiGraph {
    /// # Panics
    /// If the graph doesn't have any edges or nodes.
    pub fn from_file(file: &Path) -> io::Result<Self> {
        let mut lines = io::BufReader::new(File::open(file)?).lines();
        let num_nodes: usize = read(lines.next(), "the number of nodes")?;
        let num_edges: usize = read(lines.next(), "the number of edges")?;
        Self::check_nodes_edges_not_0(num_nodes, num_edges)?;
        let mut edges = Matrix::<EdgeCount>::new(num_nodes);

        for line in lines {
            let splitted = line?
                .split_ascii_whitespace()
                .map(|s| parse::<usize>(s, "a node number"))
                .collect::<Result<Vec<usize>, _>>()?;
            Self::check_exactly_2_numbers(&splitted)?;
            Self::check_node_ref_in_range(&splitted, num_nodes)?;
            edges[[splitted[0], splitted[1]]] += 1;
        }

        Self::check_self_loops(&edges)?;

        let cum_row_sums: Vec<usize> = (0..num_nodes).scan(0, |acc, row| {
                *acc += edges.row_iter(row).map(|v| *v as usize).sum::<usize>();
                Some(*acc)
        }).collect();
        // NOT the number of edges (but twice the number of edges)
        let &total_sum = cum_row_sums.last().unwrap();
        let result = MultiGraph {
            edges,
            cum_row_sums,
            rng: rand::thread_rng(),
            node_to_row: (0..num_nodes).collect(),
        };
        Self::check_num_edges_right(num_edges, total_sum)?;
        Ok(result)
    }

    fn check_self_loops(m: &Matrix<EdgeCount>) -> io::Result<()> {
        if m.diag_iter().sum::<u32>() > 0 {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Self loops are not supported.",
            ))
        } else {
            Ok(())
        }
    }

    fn check_nodes_edges_not_0(num_nodes: usize, num_edges: usize) -> io::Result<()> {
        if num_nodes == 0 || num_edges == 0 {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "The graph needs to have nodes and edges.",
            ))
        } else {
            Ok(())
        }
    }

    fn check_exactly_2_numbers(splitted: &Vec<usize>) -> io::Result<()> {
        if splitted.len() != 2 {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Expected exactly two numbers on each \
                 line starting from the third.",
            ))
        } else {
            Ok(())
        }
    }

    fn check_node_ref_in_range(splitted: &Vec<usize>, num_nodes: usize) -> io::Result<()> {
        if splitted[0] >= num_nodes || splitted[1] >= num_nodes {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Node number too large.",
            ))
        } else {
            Ok(())
        }
    }

    fn check_num_edges_right(num_edges: usize, total_sum: usize) -> io::Result<()> {
        if num_edges != total_sum / 2 {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "The provided number of edges is wrong.",
            ))
        } else {
            Ok(())
        }
    }

    pub fn num_edges_current(&self) -> usize {
        self.edges.stored_entries().map(|v| *v as usize).sum()
    }

    pub fn num_nodes_current(&self) -> usize {
        self.edges.dimension()
    }

    pub fn num_nodes_original(&self) -> usize {
        self.node_to_row.len()
    }

    /// `n1` and `n2` must be internal indexes (can be different from original
    /// indexes after contraction).
    pub fn num_edges_between(&self, n1: usize, n2: usize) -> usize {
        self.edges[[n1, n2]] as usize
    }

    /// The coordinates returned don't necessarily correspond to the original
    /// node numbers, because contracing edges changes the internal
    /// representation.
    /// Use `original_nodes_of(num)` to find out which node(s) in the original
    /// graph are represented by an internal node index.
    pub fn random_edge(&mut self) -> [usize; 2] {
        // random offset between 1 and (including) the cumulative sum of all rows
        let offset = self.rng.gen_range(1, *self.cum_row_sums.last().unwrap() + 1);
        let row = self
            .cum_row_sums
            .iter()
            // Find the first position in the cumulative sum that is bigger than
            // our random number.
            .position(|&e| e >= offset)
            // Note that this cannot panic because the distribution only goes up
            // to the larges element in cum_row_sums.
            .unwrap();
        let prev_rows_sum = if row >= 1 { self.cum_row_sums[row - 1] } else { 0 };
        let col_offset = offset - prev_rows_sum;
        let col = self
            .edges
            .row_iter(row)
            .scan(0, |acc, &el| {
                *acc += el as usize;
                Some(*acc)
            })
            .position(|partial_sum| partial_sum >= col_offset)
            // This unwrap cannot panic since we know that we are in a row
            // with a cumulative sum at least as big as offset and we subtracted
            // all the other rows, so the cumulative sum of the last row is at
            // least as big as col_offset.
            .unwrap();
        [row, col]
    }

    /// Returns which nodes in the original graph an internal node index
    /// represents.
    pub fn original_nodes_of(&self, internal: usize) -> Vec<usize> {
        self.node_to_row.iter()
            .enumerate()
            .filter(|(_, &v)| v == internal)
            .map(|(i, _)| i)
            .collect()
    }

    /// The values of the iterator have the shape `(orig_index, new_index)`.
    pub fn original_nodes_to_current(&self) -> impl Iterator<Item=(usize,usize)> + '_ {
        self.node_to_row.iter().copied().enumerate()
    }

    /// # Panics
    /// If the edge does not exist (i.e. if the  nodes are out of bounds or
    /// if the nodes exist but aren't connected).
    /// If the edge is a self-loop (self-loops are not supported which is
    /// checked in the constructor so the presence of a self loop would indicate
    /// a bug).
    pub fn contract_edge(&mut self, edge: [usize; 2]) {
        if self.edges[edge] == 0 {
            panic!("Trying to contract a nonexistent edge.");
        }
        if edge[0] == edge[1] {
            panic!("Self-loop detected.");
        }
        let sum_target = edge[0].max(edge[1]);
        let sum_source = edge[0].min(edge[1]);
        // no need to add / move columns as well since the matrix ensures it's
        // symmetrical
        for i in 0..self.edges.dimension() {
            self.edges[[sum_target, i]] += self.edges[[sum_source, i]];
            self.edges[[sum_source, i]] = self.edges[[0, i]];
        }
        // ignore self-loops
        self.edges[[sum_target, sum_target]] = 0;
        // TODO: possible optimization: use a different data structure to store
        // the associations.
        // (but the contraction already requires linear time so that wouldn't
        //  improve the runtime asymptotically)
        for el in &mut self.node_to_row {
            if *el == sum_source {
                *el = sum_target;
            } else if *el == 0 {
                *el = sum_source;
            }
            // we're deleting row/col 0, so all indexes get reduced by one
            *el -= 1;
        }
        self.edges.delete_row_and_col_0();
        self.cum_row_sums.clear();
        self.cum_row_sums.push(self.edges.row_iter(0)
                               .map(|v| *v as usize).sum());
        for i in 1..self.edges.dimension() {
            self.cum_row_sums.push(self.cum_row_sums[i-1] + self.edges
                                   .row_iter(i)
                                   .map(|v| *v as usize)
                                   .sum::<usize>());
        }
    }
}

impl From<Matrix<EdgeCount>> for MultiGraph {
    /// Mainly useful for creating `MultiGraph`s in the unit tests.
    /// Panics in the same cases when `MultiGraph::from_file` panics or when
    /// `MultiGraph::from_file` would return an `Err`.
    fn from(m: Matrix<EdgeCount>) -> Self {
        if m.dimension() == 0 {
            panic!("No nodes.");
        }
        let cum_row_sums: Vec<usize> = (0..m.dimension()).scan(0, |acc, row| {
                *acc += m.row_iter(row).map(|v| *v as usize).sum::<usize>();
                Some(*acc)
        }).collect();
        let total_sum = *cum_row_sums.last().unwrap();
        let node_to_row = (0..m.dimension()).collect();
        MultiGraph::check_self_loops(&m).unwrap();
        if total_sum/2 == 0 {
            panic!("No edges.");
        }
        MultiGraph {
            edges: m,
            cum_row_sums,
            rng: rand::thread_rng(),
            node_to_row,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn success() -> io::Result<()> {
        let path = PathBuf::from("./test_data/graph10");
        println!("{:?}", std::fs::canonicalize(PathBuf::from(".")));
        let graph = MultiGraph::from_file(&path)?;
        println!("{:?}", graph.edges);
        Ok(())
    }

    #[test]
    fn failures() {
        let graph = MultiGraph::from_file(&PathBuf::from("./test_data/graph10_wrong_num_edges"));
        assert_eq!(graph.unwrap_err().kind(), io::ErrorKind::InvalidInput);
        let graph2 = MultiGraph::from_file(&PathBuf::from("./test_data/graph10_wrong_num_nodes"));
        assert_eq!(graph2.unwrap_err().kind(), io::ErrorKind::InvalidInput);
        let graph3 = MultiGraph::from_file(&PathBuf::from("./test_data/graph10_parse_error"));
        assert_eq!(graph3.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn random_selection() {
        let mut m = Matrix::new(3);
        m[[0, 1]] = 1;
        m[[1, 2]] = 1;
        // the matrix looks like this:
        // 0, 1, 0
        // 1, 0, 1
        // 0, 1, 0
        let mut mg = MultiGraph::from(m);
        let mut count_edge_hits = [0, 0, 0, 0];
        // if this terminates it probably works...
        while count_edge_hits.iter().find(|&&e| e == 0).is_some() {
            let random = mg.random_edge();
            println!("{:?}", random);
            match random {
                [0, 1] => count_edge_hits[0] += 1,
                [1, 0] => count_edge_hits[1] += 1,
                [1, 2] => count_edge_hits[2] += 1,
                [2, 1] => count_edge_hits[3] += 1,
                _ => panic!("not a valid edge: {:?}", random)
            }
        }
    }

    fn find_internal_index(mg: &MultiGraph, original_index: usize) -> usize {
        for i in 0..mg.num_nodes_current() {
            if mg.original_nodes_of(i).contains(&original_index) {
                return i;
            }
        }
        panic!("Lost a node.");
    }

    #[test]
    fn contract_edge() {
        let mut m = Matrix::new(3);
        m[[0, 1]] = 1;
        m[[0, 2]] = 1;
        m[[1, 2]] = 1;
        let mut mg = MultiGraph::from(m);
        println!("{:?}", mg);
        mg.contract_edge([0, 1]);
        println!("{:?}", mg);
        // before:
        //     0
        //    / \
        //   1---2
        // after (expected):
        // 0,1---2
        assert_eq!(mg.num_edges_current(), 2);
        assert_eq!(mg.original_nodes_of(0), vec![0, 1]);
        // There is only one edge left.
        let re = mg.random_edge();
        assert!(re == [0, 1] || re == [1, 0]);
        assert_eq!(mg.edges[[0, 1]], 2);
        let mut m = Matrix::new(4);
        m[[0, 1]] = 1; m[[0, 2]] = 4; m[[0, 3]] = 3;
                       m[[1, 2]] = 0; m[[1, 3]] = 2;
                                      m[[2, 3]] = 1;
        // m looks like this:
        // 0, 1, 4, 3
        // 1, 0, 0, 2
        // 4, 0, 0, 1
        // 3, 2, 1, 0
        // the graph looks like:
        //  0-------1
        //  | \     |
        // (4) (3) (2)
        //  |     \ |
        //  2-------3
        println!("{:?}", m);
        let mut mg1 = MultiGraph::from(m.clone());
        let mut mg2 = MultiGraph::from(m);
        mg1.contract_edge([1, 0]);
        println!("{:?}", mg1);
        let i01 = find_internal_index(&mg1, 0);
        assert_eq!(i01, find_internal_index(&mg1, 1));
        let orig = mg1.original_nodes_of(i01);
        assert!(orig == vec![0, 1] || orig == vec![1, 0]);
        let i3 = find_internal_index(&mg1, 3);
        let i2 = find_internal_index(&mg1, 2);
        assert_eq!(mg1.num_edges_between(i01, i3), 5);
        assert_eq!(mg1.num_edges_between(i01, i2), 4);
        assert_eq!(mg1.num_nodes_current(), 3);
        mg2.contract_edge([3, 2]);
        println!("{:?}", mg2);
        let i23 = find_internal_index(&mg2, 2);
        assert_eq!(i23, find_internal_index(&mg2, 3));
        let orig2 = mg2.original_nodes_of(i23);
        dbg!(&orig2);
        assert!(orig2 == vec![2, 3] || orig2 == vec![3, 2]);
        let i0 = find_internal_index(&mg2, 0);
        let i1 = find_internal_index(&mg2, 1);
        assert_eq!(mg2.num_edges_between(i23, i23), 0);
        assert_eq!(mg2.num_edges_between(i23, i0), 7);
        assert_eq!(mg2.num_edges_between(i0, i23), 7);
        assert_eq!(mg2.num_edges_between(i23, i1), 2);
        assert_eq!(mg2.num_nodes_current(), 3);
        // mg2 after contraction:
        //  0--------1
        //  |      /
        // (7)  (2)
        //  |  /
        // 2,3
        mg2.contract_edge([i23, i0]);
        println!("{:?}", mg2);
        let i023 = find_internal_index(&mg2, 0);
        assert_eq!(i023, find_internal_index(&mg2, 2));
        assert_eq!(i023, find_internal_index(&mg2, 3));
        let orig3 = mg2.original_nodes_of(i023);
        dbg!(&orig3);
        assert!(orig3.len() == 3);
        assert!(orig3.contains(&3));
        assert!(orig3.contains(&2));
        assert!(orig3.contains(&0));
        let i1_new = find_internal_index(&mg2, 1);
        assert_eq!(mg2.num_edges_between(i023, i1_new), 3);
        assert_eq!(mg2.num_nodes_current(), 2);
    }
}

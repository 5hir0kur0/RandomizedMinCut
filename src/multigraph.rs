use crate::matrix::Matrix;
use rand::{rngs, self, SeedableRng, Rng};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

type EdgeCount = u32;

/// `T` is the type used to store the number of edges between nodes.
/// Nodes are always represented as indexes into a matrix of `T`,
/// so nodes don't have names and are just `usize` values.
#[derive(Debug, Clone)]
pub struct MultiGraph {
    edges: Matrix<EdgeCount>,
    row_sums: Vec<usize>,
    total_sum: usize,
    rng: rngs::StdRng,
    /// Stores which nodes are "contained" in which row/col in the matrix.
    /// After contracting an edge the two nodes at the beginning and at the end
    /// are represented by just one row/col in the matrix.
    node_to_row: Vec<usize>,
}

impl MultiGraph {
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
    /// the parsing fails return an error with the error message
    /// "Expected {unit}".
    /// Calls `parse()` to do the parsing (see above).
    fn read<T: std::str::FromStr>(input: Option<io::Result<String>>, unit: &str)
        -> io::Result<T> {
        let num = input.ok_or_else(|| {
            io::Error::new(io::ErrorKind::UnexpectedEof,
                           format!("Expected {}.", unit))
        })??;
        Self::parse(&num, unit)
    }

    /// Calcualtes the cumulative row sums.
    fn cum_row_sums(&self) -> impl Iterator<Item=usize> + '_ {
        self.row_sums.iter().scan(0, |acc, row| {
                *acc += row;
                Some(*acc)
        })
    }

    /// # Panics
    /// If the graph doesn't have any edges or nodes.
    pub fn from_file(file: &Path) -> io::Result<Self> {
        let mut lines = io::BufReader::new(File::open(file)?).lines();
        let num_nodes: usize = Self::read(lines.next(), "the number of nodes")?;
        let num_edges: usize = Self::read(lines.next(), "the number of edges")?;
        Self::check_nodes_edges(num_nodes, num_edges)?;
        let mut edges = Matrix::<EdgeCount>::new(num_nodes);

        for line in lines {
            let (source, target) = Self::retrieve_source_target(&line?)?;
            Self::check_node_ref_in_range(source, target, num_nodes)?;
            // direct access is OK since we update row_sums later...
            edges[[source, target]] += 1;
        }

        Self::check_self_loops(&edges)?;

        let row_sums: Vec<usize> = (0..num_nodes).map(|row|
                edges.row_iter(row).map(|v| *v as usize).sum::<usize>()
        ).collect();
        // NOT the number of edges (but twice the number of edges)
        let total_sum = row_sums.iter().sum();
        let result = MultiGraph {
            edges,
            row_sums,
            total_sum,
            rng: rngs::StdRng::from_rng(rand::thread_rng()).unwrap(),
            node_to_row: (0..num_nodes).collect(),
        };
        Self::check_num_edges_right(num_edges, total_sum)?;
        Ok(result)
    }

    /// Helper function for the constructor.
    fn check_self_loops(m: &Matrix<EdgeCount>) -> io::Result<()> {
        if m.diag_iter().sum::<EdgeCount>() > 0 {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Self loops are not supported.",
            ))
        } else {
            Ok(())
        }
    }

    /// Helper function for the constructor.
    fn check_nodes_edges(num_nodes: usize, num_edges: usize) -> io::Result<()> {
        if num_nodes <= 1 || num_edges == 0 {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "The graph needs to have at least two nodes and must be connected.",
            ))
        } else {
            Ok(())
        }
    }

    /// Helper function for the constructor.
    fn retrieve_source_target(line: &str) -> io::Result<(usize, usize)> {
        let mut splitted = line
            .split_ascii_whitespace()
            .map(|s| Self::parse::<usize>(s, "a node number"));
        let source = splitted.next();
        let target = splitted.next();
        if splitted.next().is_some() {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Expected exactly two numbers on each \
                 line starting from the third.",
            ))
        } else {
            match (source, target) {
                (Some(source), Some(target)) => Ok((source?, target?)),
                _ => Err(io::Error::new(io::ErrorKind::InvalidInput,
                "Expected exactly two numbers on each \
                 line starting from the third.",
            ))
            }
        }
    }

    /// Helper function for the constructor.
    fn check_node_ref_in_range(source: usize, target: usize, num_nodes: usize)
        -> io::Result<()> {
        if source >= num_nodes || target >= num_nodes {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Node number too large.",
            ))
        } else {
            Ok(())
        }
    }

    /// Helper function for the constructor.
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

    /// Current number of edges in the graph.
    pub fn num_edges_current(&self) -> usize {
        // self loops aren't allowed so this should work
        // (otherwise you'd have to "exclude" them from the `/2`)
        self.cum_row_sums().nth(self.row_sums.len() - 1).unwrap() / 2
    }

    /// Current number of nodes in the graph.
    pub fn num_nodes_current(&self) -> usize {
        self.edges.dimension()
    }

    /// Number of nodes that were in the graph when it was created.
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
        let offset = self.rng.gen_range(1, self.total_sum + 1);
        let row = self
            .cum_row_sums()
            // Find the first position in the cumulative sum that is bigger than
            // our random number.
            .position(|e| e >= offset)
            // Note that this cannot panic because the distribution only goes up
            // to the larges element in cum_row_sums (== total_sum).
            .unwrap();
        // TODO: potential optimization: optimize away the 2nd call to
        // cum_row_sums()
        // The unwrap() cannot panic since we calculated row using
        // cum_row_sums() above and the row sums cannot have changed since then.
        // The `0..1.chain()` just adds a 0 at the front of the iterator.
        let prev_rows_sum = (0..1).chain(self.cum_row_sums()).nth(row).unwrap();
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

    /// Gives the edge coordinates (only unique ones with `source <= target`)
    /// and the multiplicity of the edges.
    /// Note that `source` and `target` are internal coordinates which might be
    /// different from the original node numbers if a contraction has been
    /// performed.
    pub fn edges_iter(&self) -> impl Iterator<Item=([usize; 2], usize)> + '_ {
        self.edges.stored_entries_with_coordinates()
            .filter(|(_, &count)| count > 0)
            .map(|(coord, count)| (coord, *count as usize))
    }

    /// Returns the neighbors of the node.
    /// This method takes linear time in the number of nodes.
    pub fn neighbors_of(&self, node: usize) -> impl Iterator<Item=usize> + '_ {
        self.edges.row_iter(node).enumerate()
            .filter(|(_, &val)| val > 0)
            .map(|(i, _)| i)
    }

    /// This function should always be used to assign to `self.edges` because
    /// it makes sure that `self.cum_row_sums` stays updated.
    fn assign_edge_count(&mut self, row: usize, col: usize, new_value: EdgeCount) {
        let prev_value = self.edges[[row, col]];
        let diff = new_value as isize - prev_value as isize;
        self.row_sums[row] = (self.row_sums[row] as isize + diff) as usize;
        self.row_sums[col] = (self.row_sums[col] as isize + diff) as usize;
        self.total_sum = (self.total_sum as isize + diff) as usize;
        if row != col { // if not on diagonal it needs to be added twice
            self.total_sum = (self.total_sum as isize + diff) as usize;
        }
        self.edges[[row, col]] = new_value;
    }

    /// Calls `self.edges.delete_row_and_col_0()` but updates `self.row_sums`
    /// and `self.total_sum`.
    fn delete_row_and_col_0(&mut self) {
        let sum_0 = self.row_sums[0];
        let mut new_total = 0;
        for i in 1..self.row_sums.len() {
            self.row_sums[i - 1] = self.row_sums[i] - self.edges[[i, 0]] as usize;
            new_total += self.row_sums[i - 1];
        }
        let _ = self.row_sums.pop();
        // self.total_sum -= 2 * sum_0 + self.edges[[0, 0]] as usize;
        self.total_sum = new_total;
        self.edges.delete_row_and_col_0();
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
            // `self.edges[[sum_target, i]] += self.edges[[sum_source, i]];`
            self.assign_edge_count(sum_target, i,
                                   self.edges[[sum_target, i]]
                                   + self.edges[[sum_source, i]]);
            // `self.edges[[sum_source, i]] = self.edges[[0, i]];`
            self.assign_edge_count(sum_source, i, self.edges[[0, i]]);
        }
        // ignore self-loops
        self.assign_edge_count(sum_target, sum_target, 0);
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
        self.delete_row_and_col_0();
        // self.row_sums is updated automatically by self.assign_edge_count.
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
        let row_sums: Vec<usize> = (0..m.dimension()).map(|row|
                m.row_iter(row).map(|v| *v as usize).sum::<usize>()
        ).collect();
        let total_sum = row_sums.iter().sum();
        let node_to_row = (0..m.dimension()).collect();
        MultiGraph::check_self_loops(&m).unwrap();
        if total_sum/2 == 0 {
            panic!("No edges.");
        }
        MultiGraph {
            edges: m,
            row_sums,
            total_sum,
            rng: rngs::StdRng::from_rng(rand::thread_rng()).unwrap(),
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
        // m looks like this:
        // 0, 1, 1
        // 1, 0, 1
        // 1, 1, 0
        let mut mg = MultiGraph::from(m);
        dbg!(&mg);
        assert_eq!(mg.num_edges_current(), 3);
        println!("                                       starting contraction");
        mg.contract_edge([0, 1]);
        println!("{:?}", mg);
        println!("contraction succeeded!!");
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

        let mut m = Matrix::new(4);
        m[[0, 1]] = 1; m[[0, 2]] = 4; m[[0, 3]] = 3;
                       m[[1, 2]] = 1; m[[1, 3]] = 2;
                                      m[[2, 3]] = 1;
        let mut mg3 = MultiGraph::from(m);
        // m looks like this:
        // 0, 1, 4, 3
        // 1, 0, 1, 2
        // 4, 1, 0, 1
        // 3, 2, 1, 0
        // the graph looks like:
        //  0-------1
        //  | \  /  |
        // (4) /\  (2)
        //  | /  (3)|
        //  2-------3
        mg3.contract_edge([1, 2]);
        println!("{:?}", mg3);
        let i12 = find_internal_index(&mg3, 1);
        let i0 = find_internal_index(&mg3, 0);
        let i3 = find_internal_index(&mg3, 3);
        assert_eq!(mg3.num_edges_between(i12, i0), 5);
        assert_eq!(mg3.num_edges_between(i12, i3), 3);
        assert_eq!(mg3.num_edges_between(i0, i3), 3);
    }

    #[test]
    fn assign_edge_count() {
        let mut m = Matrix::new(4);
        m[[0, 1]] = 1; m[[0, 2]] = 4; m[[0, 3]] = 3;
                       m[[1, 2]] = 1; m[[1, 3]] = 2;
                                      m[[2, 3]] = 1;
        let mut mg = MultiGraph::from(m);
        // m looks like this:
        // 0, 1, 4, 3
        // 1, 0, 1, 2
        // 4, 1, 0, 1
        // 3, 2, 1, 0
        assert_eq!(mg.row_sums, vec![8, 4, 6, 6]);
        assert_eq!(mg.total_sum, 24);
        mg.assign_edge_count(0, 1, 42);
        // m now looks like this:
        // 0, 42, 4, 3
        // 42, 0, 1, 2
        // 4, 1, 0, 1
        // 3, 2, 1, 0
        assert_eq!(mg.edges[[1, 0]], 42);
        assert_eq!(mg.row_sums, vec![49, 45, 6, 6]);
        assert_eq!(mg.total_sum, 106);
        mg.assign_edge_count(2, 3, 5);
        // m now looks like this:
        // 0, 42, 4, 3
        // 42, 0, 1, 2
        // 4, 1, 0, 5
        // 3, 2, 5, 0
        assert_eq!(mg.edges[[3, 2]], 5);
        assert_eq!(mg.row_sums, vec![49, 45, 10, 10]);
        assert_eq!(mg.total_sum, 114);
        mg.assign_edge_count(0, 1, 8);
        // m now looks like this:
        // 0, 8, 4, 3
        // 8, 0, 1, 2
        // 4, 1, 0, 5
        // 3, 2, 5, 0
        assert_eq!(mg.edges[[1, 0]], 8);
        assert_eq!(mg.row_sums, vec![15, 11, 10, 10]);
        assert_eq!(mg.total_sum, 46);
    }

    #[test]
    fn delete_row_and_col_0() {
        let mut m = Matrix::new(4);
        m[[0, 1]] = 1; m[[0, 2]] = 4; m[[0, 3]] = 3;
                       m[[1, 2]] = 1; m[[1, 3]] = 2;
                                      m[[2, 3]] = 1;
        let mut mg = MultiGraph::from(m);
        mg.edges[[0, 0]] = 9;
        // m looks like this:
        // 9, 1, 4, 3
        // 1, 0, 1, 2
        // 4, 1, 0, 1
        // 3, 2, 1, 0
        mg.delete_row_and_col_0();
        assert_eq!(mg.row_sums, vec![3, 2, 3]);
        assert_eq!(mg.total_sum, 8);
    }
}

use crate::matrix::Matrix;
use rand::{self, distributions::Distribution};
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
    uniform_dist: rand::distributions::Uniform<usize>,
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
    fn from_file(file: &Path) -> io::Result<Self> {
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

        let cum_row_sums: Vec<usize> = (0..num_nodes)
            .scan(0, |acc, row| {
                *acc += edges.row_iter(row).map(|v| *v as usize).sum::<usize>();
                Some(*acc)
            })
            .collect();
        // NOT the number of edges (but twice the number of edges)
        let &total_sum = cum_row_sums.last().unwrap();
        let result = MultiGraph {
            edges,
            cum_row_sums,
            rng: rand::thread_rng(),
            uniform_dist: rand::distributions::Uniform::from(1..=total_sum),
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

    fn num_edges(&self) -> usize {
        self.edges.stored_entries().map(|v| *v as usize).sum()
    }

    fn random_edge(&mut self) -> [usize; 2] {
        let offset = self.uniform_dist.sample(&mut self.rng);
        let row = self
            .cum_row_sums
            .iter()
            // Find the first position in the cumulative sum that is bigger than
            // our random number.
            // Note that this cannot panic because the distribution only goes up
            // to the larges element in cum_row_sums.
            .position(|&e| e >= offset)
            .unwrap();
        let prev_rows_sum = if row >= 1 { self.cum_row_sums[row - 1] } else { 0 };
        let col_offset = offset - prev_rows_sum;
        let col = self
            .edges
            .row_iter(row)
            .scan(0, |acc, &el| {
                *acc += el as usize;
                Some(*acc)
                // This unwrap cannot panic since we know that we are in a row
                // with a cumulative sum at least as big as offset and we subtracted
                // all the other rows, so the cumulative sum of the last row is at
                // least as big as col_offset.
            })
            .position(|partial_sum| partial_sum >= col_offset)
            .unwrap();
        [row, col]
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
        let mut mg = MultiGraph {
            edges: m,
            cum_row_sums: vec![1, 3, 4],
            rng: rand::thread_rng(),
            uniform_dist: rand::distributions::Uniform::from(1..=4),
        };
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
}

use crate::matrix::Matrix;
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
    row_sums: Vec<usize>,
    total_sum: usize,
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
    fn from_file(file: &Path) -> io::Result<Self> {
        let mut lines = io::BufReader::new(File::open(file)?).lines();
        let num_nodes: usize = read(lines.next(), "the number of nodes")?;
        let num_edges: usize = read(lines.next(), "the number of edges")?;
        let mut edges = Matrix::<EdgeCount>::new(num_nodes);

        for line in lines {
            let splitted = line?
                .split_ascii_whitespace()
                .map(|s| parse::<usize>(s, "a node number"))
                .collect::<Result<Vec<usize>, _>>()?;
            if splitted.len() != 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Expected exactly two numbers on each line starting from the third.",
                ));
            }
            if splitted[0] >= num_nodes || splitted[1] >= num_nodes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Node number too large.",
                ));
            }
            edges[[splitted[0], splitted[1]]] += 1;
        }

        let row_sums: Vec<usize> = (0..num_nodes)
            .map(|row| edges.row_iter(row).map(|v| *v as usize).sum())
            .collect();
        let total_sum = row_sums.iter().sum();
        let result = MultiGraph {
            edges,
            row_sums,
            total_sum,
        };
        if num_edges != result.num_edges() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "The provided number of edges is wrong.",
            ));
        }
        Ok(result)
    }

    fn num_edges(&self) -> usize {
        let mut result = 0_usize;
        for row in 0..self.edges.dimension() {
            for col in row..self.edges.dimension() {
                result += self.edges[[row, col]] as usize;
            }
        }
        result
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
}

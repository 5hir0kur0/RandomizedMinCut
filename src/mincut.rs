use crate::multigraph::MultiGraph;
use crate::gml;
use std::collections::HashSet;
use std::io;
use std::iter::FromIterator;

#[derive(Debug)]
/// Return type for min-cut algorithms.
pub struct MinCutEstimate {
    /// One component of the cut.
    pub node_set_1: HashSet<usize>,
    /// The other nodes.
    pub node_set_2: HashSet<usize>,
    /// The number of edges that need to be "cut" to separate the two components.
    pub cut_size: usize,
    /// The `MultiGraph` that was used to construct the min-cut in the state
    /// after the min-cut algorithm ran. This is needed to generate the
    /// visualization.
    graph: MultiGraph,
}

impl MinCutEstimate {
    /// Construct a new `MinCutEstimate` from a `MultiGraph`.
    /// Note that the `MultiGraph` is expected to have only two nodes left.
    /// The original nodes corresponding to those are then stored in
    /// `node_set_1` and `node_set_2` of the `MinCutEstimate` respectively.
    /// The number of edges between these nodes is stored in `cut_size`.
    /// # Panics (in Debug)
    /// If the expectations mentioned before are not met or the `MultiGraph`
    /// violates some invariants (which should be impossible).
    fn new(graph: MultiGraph) -> Self {
        debug_assert!(graph.num_nodes_current() == 2);
        let mut node_set_1 = HashSet::new();
        let mut node_set_2 = HashSet::new();
        for (orig_index, new_index) in graph.original_nodes_to_current() {
            if new_index == 0 {
                node_set_1.insert(orig_index);
            } else if new_index == 1 {
                node_set_2.insert(orig_index);
            }
            debug_assert!(new_index == 0 || new_index == 1);
        }
        let cut_size = graph.num_edges_between(0, 1);
        debug_assert!(node_set_1.len() + node_set_2.len() == graph.num_nodes_original());
        MinCutEstimate {
            node_set_1,
            node_set_2,
            cut_size,
            graph,
        }
    }

    /// Helper function to contruct a `gml::Node`.
    fn make_node(node: usize, fill: &str) -> gml::Node {
        gml::Node {
            label: format!("{}", node),
            graphics: gml::NodeGraphics {
                fill: fill.to_string(),
            }
        }
    }

    /// Helper function to contruct a `gml::Edge`.
    fn make_edge(source: usize, target: usize, between: bool) -> gml::Edge {
        gml::Edge {
            source,
            target,
            label: "".to_string(),
            graphics: gml::EdgeGraphics {
                outline: if between {
                    "#FF0000".to_string()
                } else {
                    "#000000".to_string()
                },
                frame_thickness: if between { 2.0 } else { 1.0 },
                linemode: if between {
                    "5.0 5.0 0.0".to_string()
                } else {
                    "".to_string()
                },
            },
        }
    }

    /// Output a GML file with colored nodes and edges that indicate where the
    /// cut is.
    pub fn write_gml_visualization<W>(&self, writer: &mut W, edges: Vec<[usize; 2]>)
        -> io::Result<()>
    where
        W: io::Write,
    {
        let smaller = if self.node_set_1.len() < self.node_set_2.len() {
            &self.node_set_1
        } else {
            &self.node_set_2
        };
        let gml_nodes = self.graph.original_nodes_to_current().map(|(orig, _)| {
            if smaller.contains(&orig) {
                Self::make_node(orig, "#DA00DA")
            } else {
                Self::make_node(orig, "#00DADA")
            }
        });
            // self.node_set_1.iter()
            // .map(|&node| Self::make_node(node, "#00DADA"))
            // .chain(self.node_set_2.iter()
            //        .map(|&node| Self::make_node(node, "#DA00DA")));
        let gml_edges = edges.iter()
            .map(|&[source, target]|
                   if (self.node_set_1.contains(&source)
                   && self.node_set_2.contains(&target))
                   || (self.node_set_2.contains(&source)
                       && self.node_set_1.contains(&target)) {
                       Self::make_edge(source, target, true)
                   } else {
                       Self::make_edge(source, target, false)
                   });
        gml::write_gml(gml_nodes, gml_edges, writer)?;
        Ok(())
    }
}

/// GUESSMINCUT algorithm from page 20 in the lecture notes.
/// The graph has to be connected. You can check with `dfs`.
pub fn guess_mincut(mut mg: MultiGraph) -> MinCutEstimate {
    for _ in 0..mg.num_nodes_original() - 2 {
        let edge = mg.random_edge();
        mg.contract_edge(edge);
    }
    MinCutEstimate::new(mg)
}


/// If the graph is not connected, return a trivial mincut.
/// Otherwise return an `Err` containing the original `MultiGraph`.
pub fn check_connected(mg: MultiGraph) -> Result<MinCutEstimate, MultiGraph> {
    if let Some((cc, other)) = dfs(&mg, 0) {
        Ok(MinCutEstimate {
            node_set_1: cc,
            node_set_2: other,
            cut_size: 0,
            graph: mg,
        })
    } else {
        Err(mg)
    }
}


/// Do a depth-first search of the graph. If it has more than one connected
/// component, return the nodes of that connected component and the other nodes.
/// Otherwise return `None`.
fn dfs(mg: &MultiGraph, start: usize) -> Option<(HashSet<usize>, HashSet<usize>)> {
    let mut stack = vec![];
    let mut processed = HashSet::new();
    stack.push(start);
    while !stack.is_empty() {
        let node = stack.pop().unwrap();
        processed.insert(node);
        mg.neighbors_of(node)
            .filter(|n| !processed.contains(n))
            .for_each(|n| stack.push(n));
    }
    if processed.len() < mg.num_nodes_current() {
        let others = HashSet::from_iter(
            (0..mg.num_nodes_current())
            .filter(|n| !processed.contains(n)));
        Some((processed, others))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use crate::multigraph::MultiGraph;
    use std::path::PathBuf;

    #[test]
    fn dfs_none() {
        let mut m = Matrix::new(3);
        m[[0, 1]] = 1;
        m[[0, 2]] = 1;
        m[[1, 2]] = 1;
        let mg = MultiGraph::from(m);
        assert!(dfs(&mg, 0).is_none());

        let mut m = Matrix::new(4);
        m[[0, 1]] = 1; m[[0, 2]] = 4; m[[0, 3]] = 3;
                       m[[1, 2]] = 0; m[[1, 3]] = 2;
                                      m[[2, 3]] = 1;
        let mg = MultiGraph::from(m);
        assert!(dfs(&mg, 0).is_none());
    }

    #[test]
    fn dfs_some() {
        let mut m = Matrix::new(3);
        m[[0, 1]] = 1;
        let mg = MultiGraph::from(m);
        assert!(dfs(&mg, 0).is_some());

        let mut m = Matrix::new(5);
        m[[0, 1]] = 1; m[[0, 2]] = 4; m[[0, 3]] = 3;
                       m[[1, 2]] = 0; m[[1, 3]] = 2;
                                      m[[2, 3]] = 1;
        // the graph looks like:
        //  0-------1
        //  | \     |
        // (4) (3) (2)   4 <- isolated
        //  |     \ |
        //  2-------3
        let mg = MultiGraph::from(m);
        let (_, other) = dfs(&mg, 0).unwrap();
        let h4 = HashSet::from_iter(vec![4]);
        assert_eq!(other, h4);

        let mut m = Matrix::new(7);
        m[[0, 1]] = 1; m[[0, 2]] = 4; m[[0, 3]] = 3;
                       m[[1, 2]] = 0; m[[1, 3]] = 2;
                                      m[[2, 3]] = 1;
        m[[4, 5]] = 1; m[[5, 6]] = 1;
        // the graph looks like:
        //  0-------1
        //  | \     |
        // (4) (3) (2)   4---5---6 <- isolated
        //  |     \ |
        //  2-------3
        let mg = MultiGraph::from(m);
        let (cc, _) = dfs(&mg, 4).unwrap();
        let h456 = HashSet::from_iter(vec![4, 5, 6]);
        assert!(cc == h456);
    }

    #[test]
    fn test_check_connected() -> std::io::Result<()> {
        let path = PathBuf::from("./test_data/graph_unconnected");
        let mg = MultiGraph::from_file(&path)?;
        let mincut = check_connected(mg).unwrap();
        assert_eq!(mincut.cut_size, 0);
        let h23456 = HashSet::from_iter(2..=6);
        let h_other = HashSet::from_iter(0..=9)
            .difference(&h23456).copied().collect();
        assert!((mincut.node_set_1 == h23456 && mincut.node_set_2 == h_other)
                || (mincut.node_set_2 == h23456 && mincut.node_set_1 == h_other));
        Ok(())
    }
}

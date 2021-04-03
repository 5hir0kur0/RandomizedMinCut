use crate::multigraph::MultiGraph;
use crate::gml;
use std::collections::HashSet;
use std::io;
use std::iter::FromIterator;
use std::thread;

/// Check if the graph is a tree for graphs with at most CHECK_TREE_OPT_THRESHOLD
/// nodes. Set to 0 to disable.
/// This won't make a big difference for very dense graphs (d > 0.5).
/// It can, however, make a big difference for small densities.
/// For example, I measured more than 15x speedup with this value (compared to 0)
/// for a graph with d=0.01 and n=2000.
const CHECK_TREE_OPT_THRESHOLD: usize = 7;
/// the number of threads will be 2^THREAD_LAUNCH_THRESHOLD.
const THREAD_LAUNCH_THRESHOLD: usize = 3;

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
        debug_assert_eq!(graph.num_nodes_current(), 2);
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

/// GUESSMINCUT algorithm (randomly contracts edges)
/// The graph has to be connected. You can check with `check_connected`.
/// Also it must have at least two nodes.
pub fn guess_mincut(mut mg: MultiGraph) -> MinCutEstimate {
    for _ in 0..mg.num_nodes_original() - 2 {
        let edge = mg.random_edge();
        mg.contract_edge(edge);
    }
    MinCutEstimate::new(mg)
}

/// FASTCUT algorithm by Karger and Stein (see Figure 6 in
/// [A New Approach to the Minimum Cut Problem](https://doi.org/10.1145/234533.234534)).
/// The graph has to be connected. You can check with `check_connected`.
/// Also it must have at least two nodes.
pub fn fastcut(mg: MultiGraph) -> MinCutEstimate {
    internal_fastcut(mg, 0)
}

/// In the case of a tree, the computation of the mincut is trivial as there
/// are only a linear number of possibilies. This function just repeatedly
/// contracts the edge which represents the highest number of edges in the
/// original graph.
fn tree_optimization(mut mg: MultiGraph) -> MinCutEstimate {
    while mg.num_nodes_current() > 2 {
        let edge = mg.edges()
            .max_by_key(|&[from, to]| mg.num_edges_between(from, to))
            // This cannot fail because there are more than two nodes left and
            // we know that the graph is connected.
            .unwrap();
        mg.contract_edge(edge);
    }
    MinCutEstimate::new(mg)
}

fn internal_fastcut(mg: MultiGraph, depth: usize) -> MinCutEstimate {
    let n = mg.num_nodes_current();
    if n > 2 {
        let desired_node_num = ((n + 1) as f64 / f64::sqrt(2.0)) as usize;
        let m1;
        let m2;
        if depth >= THREAD_LAUNCH_THRESHOLD {
            // In general, the best way to make this faster would be to brute
            // force the solution for graphs smaller than a certain size.
            // I chose to optimize the case where the graph is a tree because
            // it seemed like a low-hanging fruit.
            if n <= CHECK_TREE_OPT_THRESHOLD {
                let e = mg.edges().count();
                if e + 1 == n {
                    return tree_optimization(mg);
                }
            }
            m1 = internal_fastcut(contract(mg.clone(), desired_node_num), depth);
            m2 = internal_fastcut(contract(mg, desired_node_num), depth);
        } else {
            let mg_clone = mg.clone();
            let t = thread::spawn(move ||
                internal_fastcut(contract(mg_clone, desired_node_num), depth + 1));
            m2 = internal_fastcut(contract(mg, desired_node_num), depth + 1);
            m1 = t.join().expect("Multithreading error.");
        }
        min(m1, m2)
    } else {
        // `mg` might still contain the original matrix. The matrix doesn't
        // deallocate it's storage upon row deletion so cloning might save quite
        // a lot of memory. It's cheap to do anyway since we know the matrix is
        // only 2x2 at this point.
        MinCutEstimate::new(mg.clone())
    }
}

fn min(m1: MinCutEstimate, m2: MinCutEstimate) -> MinCutEstimate {
    if m1.cut_size < m2.cut_size {
        m1
    } else {
        m2
    }
}


fn contract(mut mg: MultiGraph, desired_node_num: usize) -> MultiGraph {
    let num_nodes = mg.num_nodes_current();
    if num_nodes <= desired_node_num { return mg; }
    for _ in 0..(num_nodes - desired_node_num) {
        let re = mg.random_edge();
        mg.contract_edge(re);
    }
    mg
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
        assert_eq!(cc, h456);
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

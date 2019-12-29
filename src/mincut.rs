use crate::multigraph::MultiGraph;
use crate::gml;
use std::collections::HashSet;
use std::io;

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
pub fn guess_mincut(mut mg: MultiGraph) -> MinCutEstimate {
    for _ in 0..mg.num_nodes_original() - 2 {
        let edge = mg.random_edge();
        mg.contract_edge(edge);
    }
    MinCutEstimate::new(mg)
}

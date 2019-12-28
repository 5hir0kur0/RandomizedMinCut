use crate::multigraph::MultiGraph;
use std::collections::HashSet;

#[derive(Debug)]
pub struct MinCutEstimate {
    pub node_set_1: HashSet<usize>,
    pub node_set_2: HashSet<usize>,
    pub cut_size: usize,
}

pub fn guess_mincut(mut mg: MultiGraph) -> MinCutEstimate {
    for _ in 0..mg.num_nodes_original()-2 {
        let edge = mg.random_edge();
        mg.contract_edge(edge);
    }
    debug_assert!(mg.num_nodes_current() == 2);
    let mut node_set_1 = HashSet::new();
    let mut node_set_2 = HashSet::new();
    for (orig_index, new_index) in mg.original_nodes_to_current() {
        debug_assert!(!node_set_1.contains(&orig_index) && !node_set_2.contains(&orig_index));
        if new_index == 0 {
            node_set_1.insert(orig_index);
        } else if new_index == 1 {
            node_set_2.insert(orig_index);
        }
        debug_assert!(new_index == 0 || new_index == 1);
    }
    let cut_size = mg.num_edges_between(0, 1);
    debug_assert!(node_set_1.len() + node_set_2.len() == mg.num_nodes_original());
    MinCutEstimate {
        node_set_1,
        node_set_2,
        cut_size,
    }
}

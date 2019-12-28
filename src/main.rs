use std::env;
use std::process;
use std::path::PathBuf;

mod matrix;
mod multigraph;
mod mincut;

use multigraph::MultiGraph;

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        eprintln!("Expected a path to a graph file.");
        process::exit(1);
    }
    let path = PathBuf::from(&args[1]);
    let mg = MultiGraph::from_file(&path).expect("Failed to parse graph file.");
    let mincut = mincut::guess_mincut(mg);
    println!("{:?}", mincut);
}

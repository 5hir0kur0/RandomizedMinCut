use std::env;
use std::process;
use std::path::PathBuf;
use std::path::Path;
use std::fs::File;

mod matrix;
mod multigraph;
mod mincut;
mod gml;

use multigraph::MultiGraph;

fn main() -> std::io::Result<()> {
    let mut args = env::args().peekable();
    let _ = args.next();
    let mut visualize = false;
    let check_connected = true;
    if let Some(arg) = args.peek() {
        if arg == "-v" {
            visualize = true;
        }
    }
    if visualize { let _ = args.next(); }
    if args.peek().is_none() {
        eprintln!("Expected a path to a graph file.");
        process::exit(1);
    }
    let path_arg = args.next().unwrap();
    let path = PathBuf::from(path_arg.clone());
    let mg = MultiGraph::from_file(&path).expect("Failed to parse graph file.");
    if visualize {
        let edges = mg.edges_iter().map(|(index, count)| {
            debug_assert!(count <= 1, "count for {:?} is >= 1", index);
            index
        }).collect::<Vec<_>>();
        let out_path = unique_name(&path_arg);
        let mut file = File::create(&out_path)?;
        let mincut = mincut(mg, check_connected, mincut::fastcut);
        println!("{:?}", mincut);
        println!("writing gml to {}...", out_path);
        mincut.write_gml_visualization(&mut file, edges)?;
    } else {
        let mincut = mincut(mg, check_connected, mincut::fastcut);
        println!("{:?}", mincut);
    }
    Ok(())
}

fn unique_name(name: &str) -> String {
    if Path::new(&format!("{}.gml", name)).exists() {
        unique_name(&format!("{}_", name))
    } else {
        format!("{}.gml", name)
    }
}

fn mincut<F>(mg: MultiGraph, check_connected: bool, f: F)
    -> mincut::MinCutEstimate
where F: FnOnce(MultiGraph) -> mincut::MinCutEstimate {
    if check_connected {
        println!("Starting DFS...");
        match mincut::check_connected(mg) {
            Ok(mce) => {
                println!("Finished DFS: Graph is unconnected.");
                mce
            },
            Err(mg) => {
                println!("Finished DFS: Graph is connected.");
                println!("Starting algorithm...");
                f(mg)
            },
        }
    } else {
        f(mg)
    }
}

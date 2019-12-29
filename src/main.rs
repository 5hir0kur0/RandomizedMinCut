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
        let mincut = mincut::guess_mincut(mg);
        println!("{:?}", mincut);
        println!("writing gml to {}...", out_path);
        mincut.write_gml_visualization(&mut file, edges)?;
    } else {
        let mincut = mincut::guess_mincut(mg);
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

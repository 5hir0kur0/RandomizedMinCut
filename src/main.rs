use std::env;
use std::process;
use std::path::PathBuf;
use std::path::Path;
use std::fs::File;
use std::io;

mod matrix;
mod multigraph;
mod mincut;
mod gml;

use multigraph::MultiGraph;

fn main() -> std::io::Result<()> {
    let mut args = env::args().peekable();
    let _ = args.next();
    let mut visualize = false;
    let mut check_connected = true;
    if let Some(arg) = args.peek() {
        if arg == "-v" {
            visualize = true;
            let _ = args.next();
        } else if arg == "-s" {
            check_connected = false;
            let _ = args.next();
        } else if arg == "-vs" || arg == "-sv" {
            visualize = true;
            check_connected = false;
            let _ = args.next();
        }
    }
    if args.peek().is_none() {
        eprintln!("Expected a path to a graph file.");
        process::exit(1);
    }
    let path_arg = args.next().unwrap();
    let path = PathBuf::from(path_arg.clone());
    eprintln!("Parsing graph...");
    let mg = MultiGraph::from_file(&path).expect("Failed to parse graph file.");
    eprintln!("Finished parsing graph.");
    if visualize {
        let edges = mg.edges_iter().map(|(index, count)| {
            debug_assert!(count <= 1, "count for {:?} is >= 1", index);
            index
        }).collect::<Vec<_>>();
        let out_path = unique_name(&path_arg);
        let file = File::create(&out_path)?;
        let mut writer = io::BufWriter::new(file);
        //let (mincut, time) = mincut(mg, check_connected, mincut::guess_mincut);
        let (mincut, time) = mincut(mg, check_connected, mincut::fastcut);
        eprintln!("Finished algorithm.");
        eprintln!("{:?}", mincut);
        eprintln!("writing gml to {}...", out_path);
        mincut.write_gml_visualization(&mut writer, edges)?;
        println!("RUNTIME: {}: {}ms", path.display(), time);
        println!("CUTSIZE: {}: {}", path.display(), mincut.cut_size);
    } else {
        //let (mincut, time) = mincut(mg, check_connected, mincut::guess_mincut);
        let (mincut, time) = mincut(mg, check_connected, mincut::fastcut);
        eprintln!("{:?}", mincut);
        eprintln!("Finished algorithm.");
        println!("RUNTIME: {:?}: {}ms", path, time);
        println!("CUTSIZE: {}: {}", path.display(), mincut.cut_size);
    }
    Ok(())
}

/// Append `_` to `name` while `{name}.gml` exists.
fn unique_name(name: &str) -> String {
    if Path::new(&format!("{}.gml", name)).exists() {
        unique_name(&format!("{}_", name))
    } else {
        format!("{}.gml", name)
    }
}

/// Returns a tuple of a `MinCutEstimate` and the running time.
/// If `check_connected` is true the algorithm is not run if the graph is
/// unconnected (which is good because it doesn't support unconnected graphs).
/// In that case `0` is returned as the run time.
/// Otherwise the run time in ms will be returned.
fn mincut<F>(mg: MultiGraph, check_connected: bool, f: F)
    -> (mincut::MinCutEstimate, usize)
where F: Fn(MultiGraph) -> mincut::MinCutEstimate {
    use std::time::Instant;
    if check_connected {
        eprintln!("Starting DFS...");
        match mincut::check_connected(mg) {
            Ok(mce) => {
                eprintln!("Finished DFS: Graph is unconnected.");
                (mce, 0)
            },
            Err(mg) => {
                eprintln!("Finished DFS: Graph is connected.");
                let start = Instant::now();
                let res = run_until_no_improvements(f, mg, 5);
                (res, start.elapsed().as_millis() as usize)
            },
        }
    } else {
        let start = Instant::now();
        let res = run_until_no_improvements(f, mg, 5);
        (res, start.elapsed().as_millis() as usize)
    }
}

fn run_until_no_improvements<F>(f: F, mg: MultiGraph, rounds: usize)
    -> mincut::MinCutEstimate
where F: Fn(MultiGraph) -> mincut::MinCutEstimate
{
    let mut iters = 1;
    eprintln!("Starting round {}...", iters);
    let mut best = f(mg.clone());
    eprintln!(" current best: {}", best.cut_size);
    loop {
        let mut stop = true;
        for _ in 0..rounds {
            iters += 1;
            eprintln!("Starting round {}...", iters);
            let mut new_mg = mg.clone();
            new_mg.new_rng();
            let res = f(new_mg);
            eprintln!(" new size: {}", res.cut_size);
            if res.cut_size < best.cut_size {
                stop = false;
                best = res;
                eprintln!(" current best: {}", best.cut_size);
                break;
            }
        }
        if stop {
            println!("ITERATIONS: {}", iters);
            return best;
        }
    }
}

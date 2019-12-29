//! Quick-and-dirty hack to write GML files...

use std::io;

#[derive(Debug)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub label: String,
    pub graphics: EdgeGraphics,
}

#[derive(Debug)]
pub struct EdgeGraphics {
    pub outline: String,
    pub frame_thickness: f64,
    pub linemode: String,
}

#[derive(Debug)]
pub struct Node {
    pub label: String,
    pub graphics: NodeGraphics,
}

#[derive(Debug)]
pub struct NodeGraphics {
    pub fill: String,
}

pub fn write_gml<W, N, E>(nodes: N, edges: E, writer: &mut W) -> io::Result<()>
where
    N: Iterator<Item=Node>,
    E: Iterator<Item=Edge>,
    W: io::Write
{
    writeln!(writer, "# made with rust")?;
    block("graph", writer, 0, |writer, indent| {
        writeln!(writer, "{:indent$}directed 0", "", indent=indent)?;
        for (id, node) in nodes.enumerate() {
            // for some reason it seems like IDs have to start at 1
            let id = id + 1;
            block("node", writer, indent, |writer, indent| {
                writeln!(writer, "{:indent$}label \"{label}\"", "",
                         indent=indent,
                         label=node.label)?;
                writeln!(writer, "{:indent$}id {id}", "",
                         indent=indent,
                         id=id)?;
                block("graphics", writer, indent, |writer, indent| {
                    writeln!(writer, "{:indent$}fill \"{fill}\"", "",
                             indent=indent,
                             fill=node.graphics.fill)?;
                    Ok(())
                })?;
                Ok(())
            })?;
        }
        for (id, edge) in edges.enumerate() {
            // for some reason it seems like IDs have to start at 1
            let id = id + 1;
            block("edge", writer, indent, |writer, indent| {
                writeln!(writer, "{:indent$}id {id}", "",
                         indent=indent,
                         id=id)?;
                writeln!(writer, "{:indent$}label \"{label}\"", "",
                         indent=indent,
                         label=edge.label)?;
                writeln!(writer, "{:indent$}source {source}", "",
                         indent=indent,
                         source=edge.source + 1)?; // see above
                writeln!(writer, "{:indent$}target {target}", "",
                         indent=indent,
                         target=edge.target + 1)?; // see above
                block("graphics", writer, indent, |writer, indent| {
                    writeln!(writer, "{:indent$}outline \"{outline}\"", "",
                             indent=indent,
                             outline=edge.graphics.outline)?;
                    writeln!(writer, "{:indent$}linemode \"{linemode}\"", "",
                             indent=indent,
                             linemode=edge.graphics.linemode)?;
                    writeln!(writer,
                             "{:indent$}frameThickness {frame_thickness:.1}",
                             "", indent=indent,
                             frame_thickness=edge.graphics.frame_thickness)?;
                    Ok(())
                })?;
                Ok(())
            })?;
        }
        Ok(())
    })?;
    Ok(())
}

/// Create an indented GMl block.
/// The content of the block is supplied by the `content` closure.
fn block<W, F>(name: &str, writer: &mut W, indent: usize, content: F) -> io::Result<()>
where W: io::Write,
      F: FnOnce(&mut W, usize) -> io::Result<()>
{
    writeln!(writer, "{:indent$}{name} [", "", indent=indent, name=name)?;
    content(writer, indent + 4)?;
    writeln!(writer, "{:indent$}]", "", indent=indent)?;
    Ok(())
}

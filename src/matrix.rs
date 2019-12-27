use std::ops::IndexMut;
use std::ops::Index;
use std::fmt;

/// Store a quadratic symmetrical matrix.
pub struct Matrix<T: Default + Clone> {
    /// Expoloit that the matrix is symmetrical and only store half of it.
    /// Only the entries with row <= column are stored.
    upper_right: Vec<T>,
    num_nodes: usize,
}

impl<T> fmt::Debug for Matrix<T>
    where T: fmt::Display + Default + Clone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.num_nodes {
            write!(f, "{}", self[[row, 0]])?;
            for col in 1..self.num_nodes {
                write!(f, ", {}", self[[row, col]])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<T: Default + Clone> Matrix<T> {
    pub fn new(num_nodes: usize) -> Matrix<T> {
        Matrix {
            upper_right: vec![T::default(); num_entries(num_nodes)],
            num_nodes,
        }
    }
}

fn num_entries(num_nodes: usize) -> usize {
    // We store every diagonal from the top right corner until the middle one.
    // So the number of entries is 1 + 2 + ... + n = n * (n+1) / 2.
    (num_nodes * (num_nodes + 1)) / 2
}

/// Calculates the index in a one-dimensional vector representing the
/// two-dimensional symmetrical matrix and storing only the upper right half of
/// it. `row` must be smaller than or equal to `col`.
fn upper_right_index([row, col]: [usize; 2], num_nodes: usize) -> usize {
    debug_assert!(row <= col);
    // The first row is full, after that each row has one element less.
    // So in each row there are exactly the row number of elements "missing".
    // To get to the first element of our row, we need to subtract 0 + 1 + ...
    // + (row - 1).
    let first_of_row = if row == 0 {
        0
    } else {
        row * num_nodes - (row * (row - 1)) / 2
    };
    // At the beginning of the row there are `row` elements missing.
    // So unless we are in row zero, this value will actually point to an
    // element outside of the current row.
    let beginning_of_row = first_of_row - row;
    // Since we know that col >= row, adding col wil get us back into "our" row
    // at exactly the right position.
    beginning_of_row + col
}

/// Calculate the index for the specified row and column in the vector
/// representing the matrix.
/// # Panics
/// Panics if `row >= num_nodes` or `col >= num_nodes`.
fn calculate_index([row, col]: [usize; 2], num_nodes: usize) -> usize {
    if row >= num_nodes || col >= num_nodes {
        panic!("Index out of range: [{}, {}] (Matrix is size {2}x{2})", row,
               col, num_nodes);
    }
    if row <= col {
        upper_right_index([row, col], num_nodes)
    } else {
        // Since the matrix is symmetrical we can just swap the dimensions for
        // the lower left half.
        upper_right_index([col, row], num_nodes)
    }
}

impl<T: Default + Clone> Index<[usize; 2]> for Matrix<T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.upper_right[calculate_index(index, self.num_nodes)]
    }
}

impl<T: Default + Clone> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.upper_right[calculate_index(index, self.num_nodes)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_access() {
        let mut m: Matrix<u32> = Matrix::new(100);
        m[[1,2]] = 12;
        assert_eq!(m[[1,2]], 12);
        assert_eq!(m[[2,1]], 12);
        // make sure they are actually stored in the same memory location
        assert_eq!(&m[[1,2]] as *const _, &m[[2,1]] as *const _);
        m[[42,24]] = 4224;
        assert_eq!(m[[42,24]], 4224);
        assert_eq!(m[[24,42]], 4224);
        // check default value
        assert_eq!(m[[99,99]], 0);

        m[[0,99]] = 1099;
        assert_eq!(m[[99,0]], 1099);
    }

    #[test]
    fn completely_test_small_matrices() {
        let mut m1x1 = Matrix::new(1);
        m1x1[[0,0]] = 42;
        println!("{:?}", m1x1);
        assert_eq!(m1x1[[0,0]], 42);

        let mut m2x2 = Matrix::new(2);
        m2x2[[0, 0]] = 42;
        m2x2[[1, 0]] = 43;
        m2x2[[1, 1]] = 44;
        println!("{:?}", m2x2);
        assert_eq!(m2x2[[0,0]], 42);
        assert_eq!(m2x2[[0,1]], 43);
        assert_eq!(m2x2[[1,1]], 44);

        let mut m5x5 = Matrix::new(5);
        for row in 0..5 {
            for col in row..5 {
                dbg!(row * 10 + col);
                m5x5[[row, col]] = row * 10 + col;
            }
        }
        println!("{:?}", m5x5);
        for col in 0..5 {
            for row in col..5 {
                assert_eq!(m5x5[[row, col]], col * 10 + row);
            }
        }
    }

    #[test]
    #[should_panic]
    fn fail_empty() {
        let m = Matrix::<()>::new(0);
        m[[0,0]]
    }

    #[test]
    #[should_panic]
    fn fail_out_of_range1() {
        let m: Matrix<()> = Matrix::new(123);
        m[[123, 0]]
    }

    #[test]
    #[should_panic]
    fn fail_out_of_range2() {
        let m: Matrix<()> = Matrix::new(123);
        m[[0, 123]]
    }
}

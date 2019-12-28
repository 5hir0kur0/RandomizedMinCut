use std::ops::IndexMut;
use std::ops::Index;
use std::fmt;

/// Store a quadratic symmetrical matrix.
pub struct Matrix<T: Default + Clone> {
    /// Expoloit that the matrix is symmetrical and only store half of it.
    /// Only the entries with row <= column are stored.
    upper_right: Vec<T>,
    num_rows: usize,
    /// Needed for the deletion support.
    /// The deletion doesn't allocate or free memory.
    offset: usize,
}

impl<T: Default + Clone> Matrix<T> {
    /// Create a new symmetrical matrix of size `num_rows x num_rows`.
    pub fn new(num_rows: usize) -> Matrix<T> {
        Matrix {
            upper_right: vec![T::default(); num_entries(num_rows)],
            num_rows,
            offset: 0,
        }
    }

    /// Returns the number of rows (or columns) in the matrix.
    pub fn dimension(&self) -> usize {
        self.num_rows
    }

    /// Return an iterator over a single row in the matrix.
    /// Since the matrix is symmetrical you can just as well treat
    /// this as a column iterator.
    pub fn row_iter(&self, row: usize) -> impl Iterator<Item=&T> {
        RowIter {
            matrix: &self,
            col: 0,
            row,
        }
    }

    /// Iterator over the diagonal of the matrix.
    /// [[0, 0]], [[1, 1]], ..., [[n, n]]
    pub fn diag_iter(&self) -> impl Iterator<Item=&T> {
        DiagIter {
            matrix: &self,
            pos: 0,
        }
    }

    /// Iterator over the entries that are actually stored.
    /// I.e. entries that are the same as stored entries for symmetrical reasons
    /// won't be in the iterator.
    pub fn stored_entries(&self) -> impl Iterator<Item=&T> {
        self.upper_right[self.offset..].iter()
    }

    /// # Panics
    /// If there is no row/col 0 (i.e. if the matrix is a 0x0 matrix).
    pub fn delete_row_and_col_0(&mut self) {
        if self.num_rows == 0 {
            panic!("Trying to delete from empty matrix.");
        }
        self.offset += self.num_rows;
        self.num_rows -= 1;
    }
}

impl<T> fmt::Debug for Matrix<T>
    where T: fmt::Display + Default + Clone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.num_rows {
            write!(f, "{}", self[[row, 0]])?;
            for col in 1..self.num_rows {
                write!(f, ", {}", self[[row, col]])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<T> Clone for Matrix<T> where T: Default + Clone {
    /// Clone the matrix.
    /// Note that this does *not* copy the deleted rows and colums, but since
    /// those are not accessible through the public interface of `Matrix` anyway
    /// this is not a problem (but saves a lot of memory).
    fn clone(&self) -> Self {
        Matrix {
            upper_right: self.upper_right[self.offset..].to_vec(),
            offset: 0,
            num_rows: self.num_rows,
        }
    }
}

/// Number of entries stored in the matrix (NOT including the ones that are not
/// stored for symmetrical reasons)
fn num_entries(num_rows: usize) -> usize {
    // We store every diagonal from the top right corner until the middle one.
    // So the number of entries is 1 + 2 + ... + n = n * (n+1) / 2.
    (num_rows * (num_rows + 1)) / 2
}

/// Calculates the index in a one-dimensional vector representing the
/// two-dimensional symmetrical matrix and storing only the upper right half of
/// it. `row` must be smaller than or equal to `col`.
fn upper_right_index([row, col]: [usize; 2], num_rows: usize) -> usize {
    debug_assert!(row <= col);
    // The first row is full, after that each row has one element less.
    // So in each row there are exactly the row number of elements "missing".
    // To get to the first element of our row, we need to subtract 0 + 1 + ...
    // + (row - 1).
    let first_of_row = if row == 0 {
        0
    } else {
        row * num_rows - (row * (row - 1)) / 2
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
/// The `offset` is added to the result.
/// # Panics
/// Panics if `row >= num_rows` or `col >= num_rows`.
fn calculate_index([row, col]: [usize; 2], num_rows: usize, offset: usize)
    -> usize {
    if row >= num_rows || col >= num_rows {
        panic!("Index out of range: [{}, {}] (Matrix is size {2}x{2})", row,
               col, num_rows);
    }
    offset + if row <= col {
        upper_right_index([row, col], num_rows)
    } else {
        // Since the matrix is symmetrical we can just swap the dimensions for
        // the lower left half.
        upper_right_index([col, row], num_rows)
    }
}

impl<T: Default + Clone> Index<[usize; 2]> for Matrix<T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.upper_right[calculate_index(index, self.num_rows, self.offset)]
    }
}

impl<T: Default + Clone> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.upper_right[calculate_index(index, self.num_rows, self.offset)]
    }
}

struct RowIter<'a, T: Default + Clone> {
    matrix: &'a Matrix<T>,
    col: usize,
    row: usize,
}

impl<'a, T: Default + Clone> Iterator for RowIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.col < self.matrix.num_rows {
            let res = &self.matrix[[self.row, self.col]];
            self.col += 1;
            Some(res)
        } else {
            None
        }
    }
}

struct DiagIter<'a, T: Default + Clone> {
    matrix: &'a Matrix<T>,
    pos: usize,
}

impl<'a, T: Default + Clone> Iterator for DiagIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.matrix.num_rows {
            let res = &self.matrix[[self.pos, self.pos]];
            self.pos += 1;
            Some(res)
        } else {
            None
        }
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
    fn test_small_matrices() {
        let mut m1x1 = Matrix::new(1);
        m1x1[[0,0]] = 42;
        println!("{:?}", m1x1);
        assert_eq!(m1x1[[0,0]], 42);
        assert_eq!(m1x1.row_iter(0).sum::<u32>(), 42);
        assert_eq!(m1x1.diag_iter().sum::<u32>(), 42);

        let mut m2x2 = Matrix::new(2);
        m2x2[[0, 0]] = 42;
        m2x2[[1, 0]] = 43;
        m2x2[[1, 1]] = 44;
        println!("{:?}", m2x2);
        assert_eq!(m2x2[[0,0]], 42);
        assert_eq!(m2x2[[0,1]], 43);
        assert_eq!(m2x2[[1,1]], 44);
        let mut it = m2x2.row_iter(0);
        assert_eq!(*it.next().unwrap(), 42);
        assert_eq!(*it.next().unwrap(), 43);
        assert_eq!(it.next(), None);
        let mut it2 = m2x2.row_iter(1);
        assert_eq!(*it2.next().unwrap(), 43);
        assert_eq!(*it2.next().unwrap(), 44);
        assert_eq!(it2.next(), None);
        let mut it3 = m2x2.diag_iter();
        assert_eq!(*it3.next().unwrap(), 42);
        assert_eq!(*it3.next().unwrap(), 44);
        assert_eq!(it3.next(), None);

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
        let mut it4 = m5x5.diag_iter();
        for i in 0..5 {
            assert_eq!(*it4.next().unwrap(), i * 10 + i);
        }
        assert_eq!(it4.next(), None);
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

    #[test]
    #[should_panic]
    fn test_delete_empty() {
        let mut m: Matrix<()> = Matrix::new(0);
        m.delete_row_and_col_0()
    }

    #[test]
    fn test_delete() {
        let mut m: Matrix<i32> = Matrix::new(3);
        m[[0, 1]] = 1;
        m[[0, 2]] = 2;
        m[[1, 2]] = 3;
        // the matrix looks like this:
        // 0, 1, 2
        // 1, 0, 3
        // 2, 3, 0
        m.delete_row_and_col_0();
        dbg!(&m);
        assert_eq!(m.dimension(), 2);
        assert_eq!(m.offset, 3);
        assert_eq!(m.stored_entries().copied().collect::<Vec<i32>>(), vec![0, 3, 0]);
        assert_eq!(m[[0, 0]], 0);
        assert_eq!(m[[1, 1]], 0);
        assert_eq!(m[[0, 1]], 3);
        assert_eq!(m[[1, 0]], 3);

        let mut m2: Matrix<u8> = Matrix::new(100);
        for i in 0..100 {
            m2[[i, 0]] = i as u8 + 1;
        }
        assert_eq!(m2.row_iter(0).copied().collect::<Vec<_>>(),
                   (1..=100).collect::<Vec<_>>());
        for i in 0..100 {
            m2[[i, 1]] = 100 + i as u8 + 1;
        }
        assert_eq!(m2.row_iter(1).copied().collect::<Vec<_>>(),
                   (101..=200).collect::<Vec<_>>());
        m2.delete_row_and_col_0();
        assert_eq!(m2.row_iter(0).copied().collect::<Vec<_>>(),
                   (102..=200).collect::<Vec<_>>());
    }
}

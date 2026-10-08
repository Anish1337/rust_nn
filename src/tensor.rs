//! Owned 2D matrix.


use std::fmt;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Debug)]
pub struct Tensor {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let len = rows.checked_mul(cols).expect("tensor size overflows usize");
        Self {
            rows,
            cols,
            data: vec![0.0; len],
        }
    }

    /// Build a matrix from one slice per row. `AsRef<[f64]>` lets the caller
    /// pass arrays or slices without copying them into a common container first.
    pub fn from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Self {
        let nrows = rows.len();
        let ncols = rows.first().map(|row| row.as_ref().len()).unwrap_or(0);
        let mut data = Vec::with_capacity(nrows.saturating_mul(ncols));
        for row in rows {
            let row = row.as_ref();
            assert!(row.len() == ncols, "tensor rows must have equal length");
            data.extend_from_slice(row);
        }
        Self {
            rows: nrows,
            cols: ncols,
            data,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result[(col, row)] = self[(row, col)];
            }
        }
        result
    }

    /// Sum down the batch axis. The result is `1 × cols`.
    #[must_use]
    pub fn sum_rows(&self) -> Self {
        let mut result = Self::zeros(1, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result[(0, col)] += self[(row, col)];
            }
        }
        result
    }

    /// Add one `1 × cols` vector to every row.
    pub fn add_row_vector(&mut self, row: &Tensor) {
        assert!(
            row.rows == 1 && row.cols == self.cols,
            "invalid row vector shape"
        );
        for r in 0..self.rows {
            for c in 0..self.cols {
                self[(r, c)] += row[(0, c)];
            }
        }
    }

    #[must_use]
    pub fn matmul(&self, rhs: &Tensor) -> Self {
        assert!(
            self.cols == rhs.rows,
            "invalid dimensions for matrix multiplication"
        );
        let mut result = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    fn offset(&self, row: usize, col: usize) -> usize {
        assert!(
            row < self.rows && col < self.cols,
            "tensor index out of bounds: ({row}, {col}) for {}x{}",
            self.rows,
            self.cols
        );
        row * self.cols + col
    }

    fn check_same_shape(&self, rhs: &Tensor) {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "tensor shape mismatch: {}x{} vs {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );
    }
}

impl Index<(usize, usize)> for Tensor {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        &self.data[self.offset(row, col)]
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        let offset = self.offset(row, col);
        &mut self.data[offset]
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        self.check_same_shape(rhs);
        for (slot, value) in self.data.iter_mut().zip(&rhs.data) {
            *slot += value;
        }
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: &Tensor) {
        self.check_same_shape(rhs);
        for (slot, value) in self.data.iter_mut().zip(&rhs.data) {
            *slot -= value;
        }
    }
}

impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, scalar: f64) {
        for value in &mut self.data {
            *value *= scalar;
        }
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        let mut out = self.clone();
        out += rhs;
        out
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        let mut out = self.clone();
        out -= rhs;
        out
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f64) -> Tensor {
        let mut out = self.clone();
        out *= scalar;
        out
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        rhs * self
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            write!(f, "[ ")?;
            for col in 0..self.cols {
                write!(f, "{:10.6} ", self[(row, col)])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

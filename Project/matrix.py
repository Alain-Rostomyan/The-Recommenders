import numpy as np
import scipy.linalg as sp
import scipy.sparse.linalg as spla
from matrix_operation import matrix_operations, sparse_matrix_operations

class Matrix(matrix_operations):  # base matrix class

    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) # or data.shape? -----

    def apply_threshold(self, threshold):
        return Matrix(np.where(self.data > threshold, 1, 0))
        
    # def apply_threshold(self, threshold):
    #     transform = lambda x:1 if x > threshold else 0

    #     transformer = np.vectorize(transform)
    #     return(transformer(self))


    # def apply_threshold(self, threshold):
    #     apply_threshold_lambda = lambda data: Matrix(np.where(data > threshold, 1, 0))
    #     return apply_threshold_lambda(self.data)
    
    @staticmethod
    def generate_random_non_square():
        rows = np.random.randint(2, 101)  # Random number of rows (2 to 100)
        columns = np.random.randint(1, rows)  # Random number of columns (1 to rows - 1)
        data = np.random.rand(rows, columns)
        return Matrix(data)


        
class DenseMatrix(Matrix):  # Child class of Matrix
    def __init__(self, data):
        super().__init__(data)

    def solve_linear_system(self, b):
        if self.rows != self.cols:
            raise ValueError("Linear system can only be solved for square matrices")

        return sp.solve(self.data, b)
    
    def apply_threshold(self, threshold):
        return DenseMatrix(np.where(self.data > threshold, 1, 0))

    # def apply_threshold(self, threshold):
    #     # Function to transform the matrix into binary where the values are 1 if the element
    #     # of the matrix is greater than or equal to the threshold and 0 otherwise
    #     threshold_func = lambda x: 1 if x >= threshold else 0
    #     self.data = np.vectorize(threshold_func)(self.data)    


class SparseMatrix(Matrix):  # Child class of Matrix
    def __init__(self, data):
        super().__init__(data)

        # Convert dense matrix to COO format
        self.row_indices = []
        self.col_indices = []
        self.values = []

        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] != 0:
                    self.values.append(self.data[i][j])
                    self.row_indices.append(i)
                    self.col_indices.append(j)
        
    def solve_linear_system(self, b):
        if self.rows != self.cols:
            raise ValueError("Linear system can only be solved for square matrices")

        if isinstance(self, SparseMatrix):
            return spla.spsolve(self.data, b)
        else:
            raise ValueError("Sparse matrix support with dense solver is not implemented")

    def mat_vec_multiplication(self, vector):
        if len(vector) != self.cols:
            raise ValueError("Matrix columns must match vector length")

        result = np.zeros(self.rows)

        for i in range(len(self.values)):
            result[self.row_indices[i]] += self.values[i] * vector[self.col_indices[i]]

        return result

    def mat_mat_multiplication(self, other_matrix):
        if self.cols != other_matrix.rows:
            raise ValueError("Number of columns in the first matrix must match number of rows in the second matrix")

        result = np.zeros((self.rows, other_matrix.cols))

        for i in range(len(self.values)):
            for j in range(other_matrix.cols):
                if self.col_indices[i] == j:
                    for k in range(other_matrix.rows):
                        result[self.row_indices[i], k] += self.values[i] * other_matrix.data[j, k]

        return result
    
    # def apply_threshold(self, threshold):
    #     p = lambda x: x > threshold , 1, 0
    #     rows_new = [p(x) for x in self.rows]
    #     cols_new = [p(x) for x in self.cols]

    def apply_threshold(self, threshold):
        transformed_indices_and_values = [
            (row, col, 1) for row, col, value in zip(self.rows, self.cols, self.values) if value > threshold
        ]

        if not transformed_indices_and_values:
            return SparseMatrix([], [], [], self.shape)

        new_rows, new_cols, new_values = zip(*transformed_indices_and_values)

        return SparseMatrix(list(new_rows), list(new_cols), list(new_values), self.shape)

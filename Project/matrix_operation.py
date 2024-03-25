import numpy as np
import scipy.linalg as sp


class matrix_operations:
    # --common methods for matrix operations
    def mat_vec_multiplication(self, vector):  # Matrix-Vector Multiplication
        if self.cols != vector.shape[0]:  # Check the number of columns in the matrix
            raise ValueError("Matrix columns must match vector length")
    
        result = np.zeros(self.rows) # Initialise result matrix

        for i in range(self.rows):
            result[i] = np.dot(self.data[i], vector) # Compute new matrix values using dot product
        return result


    def mat_mat_multiplication(self, other_matrix):  # Matrix-Matrix Multiplication
        if self.cols != other_matrix.rows:
            raise ValueError("Number of columns in the first matrix must match number of rows in the second matrix")
        
        result = np.zeros((self.rows, other_matrix.cols))  # Initialise result matrix
        
        for i in range(self.rows):
            for j in range(other_matrix.cols):
                for k in range(self.cols):
                    result[i, j] += self.data[i, k] * other_matrix.data[k, j] # Compute new matrix values
        
        return result


    def mat_addition(self, other_matrix):  # Matrix Addition
        if self.rows != other_matrix.rows or self.cols != other_matrix.cols:
            raise ValueError("Matrices must have the same dimensions for addition")
        
        result = np.zeros((self.rows, self.cols))  # Initialise result matrix
        
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self.data[i, j] + other_matrix.data[i, j]
        
        return result


    def mat_subtraction(self, other_matrix):  # Matrix Subtraction
        if self.rows != other_matrix.rows or self.cols != other_matrix.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction")
        
        result = np.zeros((self.rows, self.cols))  # Initialise result matrix
        
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self.data[i, j] - other_matrix.data[i, j]
        
        return result

    def l1_norm(self): # L1 norm is sum of absolute values of entries
        return sum(abs(self.data)) 

    def l2_norm(self): # L2 norm is squareroot of sum of entries
        return (sum(self.data ** 2)) ** (1/2)

    def linf_norm(self): # Linfinity norm is max of absolute values of entries
        return np.max(np.abs(self.data))


    def eigenvalues(self): # Compute eigenvalues of square matrices
        if self.rows != self.cols:
            raise ValueError("Eigenvalues can only be computed for square matrices")
        
        eigvals, _ = sp.eig(self.data)
        return eigvals

    def svd(self): # Perform SVD for non-square matrices
        U, S, Vh = sp.svd(self.data)
        return U, S, Vh
    
    def dimensionality_reduction_with_svd(data_matrix, num_singular_values):
        # Perform SVD
        U, S, Vh = np.linalg.svd(data_matrix)

        # Select the first num_singular_values singular values and corresponding columns of U and Vh
        U_reduced = U[:, :num_singular_values]
        S_reduced = np.diag(S[:num_singular_values])
        Vh_reduced = Vh[:num_singular_values, :]

        # Reconstruct the data matrix using the reduced matrices
        data_matrix_reconstructed = np.dot(U_reduced, np.dot(S_reduced, Vh_reduced))

        # Print the selected singular values
        print("Selected Singular Values:")
        print(S[:num_singular_values])

        # Print the values in matrices U and Vᵀ
        print("Matrix U:")
        print(U)
        print("Matrix Vᵀ:")
        print(Vh.T)

        return data_matrix_reconstructed
    
    def dimensionality_reduction_for_recommender(data_matrix, num_singular_values):
        # Perform SVD
        U, S, Vh = np.linalg.svd(data_matrix)

        # Select the first num_singular_values singular values and corresponding columns of U and Vh
        U_reduced = U[:, :num_singular_values]
        S_reduced = np.diag(S[:num_singular_values])
        Vh_reduced = Vh[:num_singular_values, :]

        # Reconstruct the data matrix using the reduced matrices
        data_matrix_reconstructed = np.dot(U_reduced, np.dot(S_reduced, Vh_reduced))

        return Vh.T    
    
class sparse_matrix_operations:
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

class dense_matrix_operations:
    pass
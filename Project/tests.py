#     @staticmethod
#     def test_norm(matrix_size):
#         dense_data = create_random_matrix(matrix_size, matrix_size)
#         sparse_data = create_random_matrix(matrix_size, matrix_size)

#         dense_matrix = DenseMatrix(dense_data)
#         sparse_matrix = SparseMatrix(sparse_data)

#         dense_norm_l1 = dense_matrix.l1_norm()
#         sparse_norm_l1 = np.linalg.norm(sparse_matrix.data, ord=1)

#         dense_norm_l2 = dense_matrix.l2_norm()
#         sparse_norm_l2 = np.linalg.norm(sparse_matrix.data, ord=2)

#         dense_norm_linf = dense_matrix.linf_norm()
#         sparse_norm_linf = np.linalg.norm(sparse_matrix.data, ord=np.inf)

#         return dense_norm_l1, sparse_norm_l1, dense_norm_l2, sparse_norm_l2, dense_norm_linf, sparse_norm_linf

#     @staticmethod
#     def test_eig(matrix_size):
#         dense_data = create_random_matrix(matrix_size, matrix_size)
#         sparse_data = create_random_matrix(matrix_size, matrix_size)

#         dense_matrix = DenseMatrix(dense_data)
#         sparse_matrix = SparseMatrix(sparse_data)

#         dense_eigvals = dense_matrix.eigenvalues()
#         sparse_eigvals = np.linalg.eigvals(sparse_matrix.data)

#         return dense_eigvals, sparse_eigvals

#     @staticmethod
#     def test_SVD(matrix_size):
#         dense_data = create_random_matrix(matrix_size, matrix_size)
#         sparse_data = create_random_matrix(matrix_size, matrix_size)

#         dense_matrix = DenseMatrix(dense_data)
#         sparse_matrix = SparseMatrix(sparse_data)

#         dense_U, dense_S, dense_Vh = dense_matrix.svd()
#         sparse_U, sparse_S, sparse_Vh = np.linalg.svd(sparse_matrix.data)

#         return dense_U, dense_S, dense_Vh, sparse_U, sparse_S, sparse_Vh

#     @staticmethod
#     def test_lin_solve(matrix_size):
#         dense_data = create_random_matrix(matrix_size, matrix_size)
#         sparse_data = create_random_matrix(matrix_size, matrix_size)

#         dense_matrix = DenseMatrix(dense_data)
#         sparse_matrix = SparseMatrix(sparse_data)

#         b = np.random.rand(matrix_size)

#         dense_solution = dense_matrix.solve_linear_system(b)
#         sparse_solution = sparse_matrix.solve_linear_system(b)

#         return dense_solution, sparse_solution



'''
1. Test addition
2. Test subtraction
3. Test mat x mat multiplcation
4. Test mat x vec multiplcation
5. Test norms
6. Test solve lin systems
7. Test compute eig
8. Test perform SVD

- Create code for plotting time taken for each of these for matrices of size 500
- Aim to get addition test to atleast 500 matrix size, ideally more
- Probably just plot time taken differences for add, mat x mult, & mat x vec

'''
from matrix import DenseMatrix, SparseMatrix
import numpy as np
import time

def create_random_matrix(rows, cols, sparsity=0.5):
    """
    Create a random matrix with the given size and sparsity.
    """
    data = np.random.rand(rows, cols)
    data[data > sparsity] = 0
    return data

def test_threshold_application(matrix_size):
    dense_data = create_random_matrix(matrix_size, matrix_size)
    sparse_data = create_random_matrix(matrix_size, matrix_size)

    dense_matrix = DenseMatrix(dense_data)
    sparse_matrix = SparseMatrix(sparse_data)

    threshold = 5

    dense_binary_matrix = dense_matrix.apply_threshold(threshold)
    sparse_binary_matrix = sparse_matrix.apply_threshold(threshold)

    print("Dense Binary Matrix:")
    print(dense_binary_matrix.data)
    print("Sparse Binary Matrix:")
    print(sparse_binary_matrix.data)


class test_matrix_operations: # All matrix operation tests
    def test_mat_mat_multiplication(matrix_size): # testing mat mat mult

        dense_data = create_random_matrix(matrix_size, matrix_size)
        sparse_data = create_random_matrix(matrix_size, matrix_size)

        dense_matrix = DenseMatrix(dense_data)
        sparse_matrix = SparseMatrix(sparse_data)

        start_time = time.time()
        _ = dense_matrix.mat_mat_multiplication(dense_matrix)
        dense_matmat_time = time.time() - start_time

        start_time = time.time()
        _ = sparse_matrix.mat_mat_multiplication(sparse_matrix)
        sparse_matmat_time = time.time() - start_time

        return dense_matmat_time, sparse_matmat_time
    
    def test_mat_vec_multiplication(matrix_size): # testing mat vec mult
        dense_data = create_random_matrix(matrix_size, matrix_size)
        sparse_data = create_random_matrix(matrix_size, matrix_size)

        dense_matrix = DenseMatrix(dense_data)
        sparse_matrix = SparseMatrix(sparse_data)

        # Create a random vector with the appropriate length
        vector = np.random.rand(matrix_size)

        start_time = time.time()
        _ = dense_matrix.mat_vec_multiplication(vector)
        dense_matmat_time = time.time() - start_time

        start_time = time.time()
        _ = sparse_matrix.mat_vec_multiplication(vector)
        sparse_matmat_time = time.time() - start_time

        return dense_matmat_time, sparse_matmat_time

    def test_mat_addition(matrix_size): # testing mat addition
        dense_data = create_random_matrix(matrix_size, matrix_size)
        sparse_data = create_random_matrix(matrix_size, matrix_size)

        dense_matrix = DenseMatrix(dense_data)
        sparse_matrix = SparseMatrix(sparse_data)

        start_time = time.time()
        _ = dense_matrix.mat_addition(dense_matrix)
        dense_matmat_time = time.time() - start_time

        start_time = time.time()
        _ = sparse_matrix.mat_addition(sparse_matrix)
        sparse_matmat_time = time.time() - start_time

        return dense_matmat_time, sparse_matmat_time
    
    def test_mat_subtraction(matrix_size): # testing mat subtraction
        dense_data = create_random_matrix(matrix_size, matrix_size)
        sparse_data = create_random_matrix(matrix_size, matrix_size)

        dense_matrix = DenseMatrix(dense_data)
        sparse_matrix = SparseMatrix(sparse_data)

        start_time = time.time()
        _ = dense_matrix.mat_subtraction(dense_matrix)
        dense_matmat_time = time.time() - start_time

        start_time = time.time()
        _ = sparse_matrix.mat_subtraction(sparse_matrix)
        sparse_matmat_time = time.time() - start_time

        return dense_matmat_time, sparse_matmat_time
    

    def test_all_operations(matrix_size): # testing all operations -- maybe not needed?
        dense_time_mat_mat_mult, sparse_time_mat_mat_mult = test_matrix_operations.test_mat_mat_multiplication(matrix_size)
        dense_time_mat_vec_mult, sparse_time_mat_vec_mult = test_matrix_operations.test_mat_vec_multiplication(matrix_size)
        dense_time_mat_add, sparse_time_mat_add = test_matrix_operations.test_mat_addition(matrix_size)
        dense_time_mat_sub, sparse_time_mat_sub = test_matrix_operations.test_mat_subtraction(matrix_size)

        return dense_time_mat_mat_mult, sparse_time_mat_mat_mult, dense_time_mat_vec_mult, sparse_time_mat_vec_mult, dense_time_mat_add, sparse_time_mat_add, dense_time_mat_sub, sparse_time_mat_sub
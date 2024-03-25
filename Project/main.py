import numpy as np
import scipy.linalg as sp
import numpy.linalg as np_linalg
import time
import matplotlib.pyplot as plt
from matrix import DenseMatrix, SparseMatrix, Matrix
import matrix_operation
from matrix_operation import matrix_operations
from tests import test_matrix_operations
from recommender import recommender

import tests


def main():
    matrix_sizes = [20, 40, 80, 125]  # Adjust this list based on your computational resources

    dense_times_mat_mat_mult = []
    sparse_times_mat_mat_mult = []
    dense_times_mat_vec_mult = []
    sparse_times_mat_vec_mult = []
    dense_times_mat_add = []
    sparse_times_mat_add = []
    dense_times_mat_sub = []
    sparse_times_mat_sub = []

    for size in matrix_sizes:
        dense_time_mat_mat_mult, sparse_time_mat_mat_mult = test_matrix_operations.test_mat_mat_multiplication(size)
        dense_time_mat_vec_mult, sparse_time_mat_vec_mult = test_matrix_operations.test_mat_vec_multiplication(size)
        dense_time_mat_add, sparse_time_mat_add = test_matrix_operations.test_mat_addition(size)
        dense_time_mat_sub, sparse_time_mat_sub = test_matrix_operations.test_mat_subtraction(size)

        dense_times_mat_mat_mult.append(dense_time_mat_mat_mult)
        sparse_times_mat_mat_mult.append(sparse_time_mat_mat_mult)
        dense_times_mat_vec_mult.append(dense_time_mat_vec_mult)
        sparse_times_mat_vec_mult.append(sparse_time_mat_vec_mult)
        dense_times_mat_add.append(dense_time_mat_add)
        sparse_times_mat_add.append(sparse_time_mat_add)
        dense_times_mat_sub.append(dense_time_mat_sub)
        sparse_times_mat_sub.append(sparse_time_mat_sub)

    # Plotting the performance comparison results
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, dense_times_mat_mat_mult, marker='o', label='Dense Matrix Mat x Mat')
    plt.plot(matrix_sizes, sparse_times_mat_mat_mult, marker='o', label='Sparse Matrix Mat x Mat')
    plt.plot(matrix_sizes, dense_times_mat_vec_mult, marker='o', label='Dense Matrix Mat x Vec')
    plt.plot(matrix_sizes, sparse_times_mat_vec_mult, marker='o', label='Sparse Matrix Mat x Vec')
    plt.plot(matrix_sizes, dense_times_mat_add, marker='o', label='Dense Matrix Mat Add')
    plt.plot(matrix_sizes, sparse_times_mat_add, marker='o', label='Sparse Matrix Mat Add')
    plt.plot(matrix_sizes, dense_times_mat_sub, marker='o', label='Dense Matrix Mat Sub')
    plt.plot(matrix_sizes, sparse_times_mat_sub, marker='o', label='Sparse Matrix Mat Sub')
    plt.title('Performance Comparison of Matrix Operations')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Adding logarithmic scale to the y-axis
    plt.show()


    # Generating and transforming non-square matrices
    non_square_matrix = Matrix.generate_random_non_square()

    threshold = 0.5  # Adjust threshold as needed
    # binary_non_square_matrix = non_square_matrix.apply_threshold(threshold)
    binary_non_square_matrix = non_square_matrix.apply_threshold(threshold)

    print("Non-square Matrix:")
    print(non_square_matrix.data)
    print("Binary Non-square Matrix:")
    print(binary_non_square_matrix.data.data)

    # Extract the NumPy array from the Matrix object
    binary_non_square_np = binary_non_square_matrix.data

    # Perform SVD using scipy.linalg.svd
    start_time = time.time()
    U_sp, S_sp, Vh_sp = sp.svd(binary_non_square_np)
    svd_time_sp = time.time() - start_time

    # Perform SVD using numpy.linalg.svd
    start_time = time.time()
    U_np, S_np, Vh_np = np_linalg.svd(binary_non_square_np)
    svd_time_np = time.time() - start_time
    S_sp = S_sp[:8]
    S_np = S_np[:8]
    
    # Compare computation time
    print("Computation Time (scipy.linalg.svd):", svd_time_sp)
    print("Computation Time (numpy.linalg.svd):", svd_time_np)

    # Plot singular values
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(S_sp) + 1), S_sp, marker='o', label='scipy.linalg.svd')
    plt.plot(range(1, len(S_np) + 1), S_np, marker='o', label='numpy.linalg.svd')
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Adding logarithmic scale to the y-axis
    plt.show()

    reduced_data_matrix = matrix_operations.dimensionality_reduction_with_svd(binary_non_square_np, 8) # change num singular values
    print(reduced_data_matrix.shape)

    # Recommender
    liked_movie_index = 0  # Index of the liked movie
    selected_movies_num = 5  # Number of recommended movies
    VT = matrix_operations.dimensionality_reduction_for_recommender(binary_non_square_np, 8)  # Example data for VT matrix

    recommended_movies = recommender.recommend(liked_movie_index, VT, selected_movies_num)
    print("Recommended Movies:", recommended_movies)
    


if __name__ == "__main__":
    main()
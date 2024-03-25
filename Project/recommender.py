import numpy as np

class recommender:
    def recommend(liked_movie_index, VT, selected_movies_num):
        recommended = []

        for i in range(VT.shape[0]):
            if i != liked_movie_index:
                dot_product = np.dot(VT[i], VT[liked_movie_index])
                recommended.append((i, dot_product))

        # Sort recommended movies based on dot product values in descending order
        recommended = sorted(recommended, key=lambda x: x[1], reverse=True)

        # Return the top selected_movies_num recommended movies
        return recommended[:selected_movies_num]
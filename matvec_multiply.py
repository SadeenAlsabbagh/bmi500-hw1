# BMI500 HW1 code review


# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function



# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function



# create a main function to test the matrix-vector product function using randomly generated data of size 1000x1000
# add comments for the selected function
import random


# Compute the dot product of two vectors using a for loop.
def dot_product(vector_a, vector_b):
    # Both inputs must be lists.
    if not isinstance(vector_a, list) or not isinstance(vector_b, list):
        raise TypeError("Both inputs must be lists.")

    # A dot product requires vectors of equal length.
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    # Empty vectors are not accepted for this implementation.
    if len(vector_a) == 0:
        raise ValueError("Vectors cannot be empty.")

    result = 0

    # Multiply corresponding elements and accumulate their sum.
    for i in range(len(vector_a)):
        if not isinstance(vector_a[i], (int, float)) or not isinstance(
            vector_b[i], (int, float)
        ):
            raise TypeError("Vector elements must be numeric.")

        result += vector_a[i] * vector_b[i]

    return result


# Compute a matrix-vector product using the dot_product function.
def matrix_vector_product(matrix, vector):
    # Matrix must be represented as a non-empty list of rows.
    if not isinstance(matrix, list):
        raise TypeError("Matrix must be a list.")

    if len(matrix) == 0:
        raise ValueError("Matrix cannot be empty.")

    if not isinstance(vector, list):
        raise TypeError("Vector must be a list.")

    if len(vector) == 0:
        raise ValueError("Vector cannot be empty.")

    result = []

    # Each row must be a list and must have the same length as the vector.
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("Each matrix row must be a list.")

        if len(row) != len(vector):
            raise ValueError(
                "Each matrix row must have the same length as the vector."
            )

        result.append(dot_product(row, vector))

    return result


# Test the matrix-vector product using randomly generated 1000x1000 data.
def main():
    size = 1000

    # Generate a 1000 x 1000 matrix with random floating-point values.
    matrix = [
        [random.random() for _ in range(size)]
        for _ in range(size)
    ]

    # Generate a vector of length 1000.
    vector = [random.random() for _ in range(size)]

    result = matrix_vector_product(matrix, vector)

    print(f"Matrix size: {size} x {size}")
    print(f"Vector size: {len(vector)}")
    print(f"Result size: {len(result)}")
    print("First five output values:", result[:5])


if __name__ == "__main__":
    main()

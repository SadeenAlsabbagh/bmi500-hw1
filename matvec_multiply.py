
# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function



# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function



# create a main function to test the matrix-vector product function using randomly generated data of size 1000x1000
# add comments for the selected function
import random


# Return True for int or float values. bool is a subclass of int in Python,
# so it is excluded explicitly to avoid treating True/False as 1/0.
def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# Compute the dot product of two vectors using a for loop.
# NaN and infinity are valid floats and propagate according to normal Python
# floating-point arithmetic; they are not treated as invalid input.
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
        if not _is_number(vector_a[i]):
            raise TypeError(
                f"vector_a element at index {i} must be an int or float."
            )

        if not _is_number(vector_b[i]):
            raise TypeError(
                f"vector_b element at index {i} must be an int or float."
            )

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
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"Matrix row {row_index} must be a list.")

        if len(row) != len(vector):
            raise ValueError(
                f"Matrix row {row_index} has length {len(row)}, but the "
                f"vector has length {len(vector)}."
            )

        # Re-raise element type errors with matrix/vector context, since
        # dot_product only knows its arguments as vector_a and vector_b.
        try:
            result.append(dot_product(row, vector))
        except TypeError as err:
            raise TypeError(
                f"Matrix row {row_index} and the vector must contain only "
                f"int or float values ({err})"
            ) from err

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

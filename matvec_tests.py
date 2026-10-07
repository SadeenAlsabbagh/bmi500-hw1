# generate tests for matrix-vector-product function in matvec_multiply.py

# generate tests for dot-product function in matvec_multiply.py

import copy
import math
import unittest

from matvec_multiply import dot_product, matrix_vector_product


class TestDotProduct(unittest.TestCase):

    def test_basic_dot_product(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)

    def test_negative_values(self):
        self.assertEqual(dot_product([-1, 2], [3, -4]), -11)

    def test_float_values(self):
        self.assertAlmostEqual(
            dot_product([1.5, 2.0], [2.0, 3.0]),
            9.0
        )

    def test_different_lengths(self):
        with self.assertRaises(ValueError):
            dot_product([1, 2], [1, 2, 3])

    def test_empty_vectors(self):
        with self.assertRaises(ValueError):
            dot_product([], [])

    def test_non_numeric_values(self):
        with self.assertRaises(TypeError):
            dot_product([1, "a"], [2, 3])

    def test_non_numeric_value_in_second_vector(self):
        with self.assertRaisesRegex(TypeError, "vector_b element at index 1"):
            dot_product([1, 2], [3, None])

    def test_non_list_input(self):
        with self.assertRaises(TypeError):
            dot_product((1, 2), [3, 4])

        with self.assertRaises(TypeError):
            dot_product([1, 2], "34")

    def test_single_element_vectors(self):
        self.assertEqual(dot_product([3], [4]), 12)

    def test_zero_values(self):
        self.assertEqual(dot_product([0, 0, 0], [1, 2, 3]), 0)
        self.assertEqual(dot_product([1, -1], [1, 1]), 0)

    def test_mixed_int_and_float_values(self):
        self.assertAlmostEqual(dot_product([1, 2.5, -3], [2.0, 4, 0.5]), 10.5)

    def test_boolean_values_rejected(self):
        # bool is a subclass of int, but True/False are not treated as 1/0.
        with self.assertRaises(TypeError):
            dot_product([True, False], [1, 2])

        with self.assertRaises(TypeError):
            dot_product([1, 2], [1, True])

    def test_nan_and_infinity_propagate(self):
        # Non-finite floats follow normal Python floating-point arithmetic.
        self.assertTrue(math.isnan(dot_product([float("nan"), 1], [1, 2])))
        self.assertEqual(dot_product([float("inf"), 1], [2, 3]), float("inf"))
        self.assertTrue(math.isnan(dot_product([float("inf")], [0])))

    def test_inputs_not_mutated(self):
        vector_a = [1, 2.5, -3]
        vector_b = [4, 0, 6.0]
        expected_a = copy.deepcopy(vector_a)
        expected_b = copy.deepcopy(vector_b)

        dot_product(vector_a, vector_b)

        self.assertEqual(vector_a, expected_a)
        self.assertEqual(vector_b, expected_b)


class TestMatrixVectorProduct(unittest.TestCase):

    def test_basic_matrix_vector_product(self):
        matrix = [
            [1, 2],
            [3, 4]
        ]
        vector = [5, 6]

        self.assertEqual(
            matrix_vector_product(matrix, vector),
            [17, 39]
        )

    def test_identity_matrix(self):
        matrix = [
            [1, 0],
            [0, 1]
        ]
        vector = [7, 8]

        self.assertEqual(
            matrix_vector_product(matrix, vector),
            [7, 8]
        )

    def test_dimension_mismatch(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        vector = [1, 2]

        with self.assertRaises(ValueError):
            matrix_vector_product(matrix, vector)

    def test_empty_matrix(self):
        with self.assertRaises(ValueError):
            matrix_vector_product([], [1, 2])

    def test_non_list_matrix(self):
        with self.assertRaises(TypeError):
            matrix_vector_product("not a matrix", [1, 2])

    def test_non_numeric_matrix_value(self):
        matrix = [
            [1, 2],
            [3, "x"]
        ]

        with self.assertRaises(TypeError):
            matrix_vector_product(matrix, [1, 2])

    def test_non_list_vector(self):
        with self.assertRaises(TypeError):
            matrix_vector_product([[1, 2]], (1, 2))

    def test_empty_vector(self):
        with self.assertRaises(ValueError):
            matrix_vector_product([[1, 2]], [])

    def test_non_list_matrix_row(self):
        with self.assertRaisesRegex(TypeError, "Matrix row 1 must be a list"):
            matrix_vector_product([[1, 2], (3, 4)], [1, 2])

    def test_ragged_matrix(self):
        matrix = [
            [1, 2],
            [3, 4, 5]
        ]

        with self.assertRaisesRegex(ValueError, "Matrix row 1 has length 3"):
            matrix_vector_product(matrix, [1, 2])

    def test_non_numeric_vector_value(self):
        with self.assertRaisesRegex(TypeError, "Matrix row 0"):
            matrix_vector_product([[1, 2], [3, 4]], [1, "b"])

    def test_non_square_matrix(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        vector = [1, 0, -1]

        self.assertEqual(matrix_vector_product(matrix, vector), [-2, -2])

    def test_single_row_matrix(self):
        self.assertEqual(matrix_vector_product([[1, 2, 3]], [4, 5, 6]), [32])

    def test_single_column_matrix(self):
        matrix = [
            [2],
            [-3],
            [0]
        ]

        self.assertEqual(matrix_vector_product(matrix, [5]), [10, -15, 0])

    def test_zero_matrix(self):
        matrix = [
            [0, 0],
            [0, 0]
        ]

        self.assertEqual(matrix_vector_product(matrix, [7, 8]), [0, 0])

    def test_mixed_int_and_float_values(self):
        matrix = [
            [1, 0.5],
            [2.0, -1]
        ]
        result = matrix_vector_product(matrix, [4, 2.0])

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 5.0)
        self.assertAlmostEqual(result[1], 6.0)

    def test_boolean_values_rejected(self):
        with self.assertRaises(TypeError):
            matrix_vector_product([[True, 1], [0, 1]], [1, 1])

        with self.assertRaises(TypeError):
            matrix_vector_product([[1, 2], [3, 4]], [False, 1])

    def test_nan_and_infinity_propagate(self):
        matrix = [
            [float("nan"), 1],
            [float("inf"), 1],
            [1, 1]
        ]
        result = matrix_vector_product(matrix, [1, 2])

        self.assertTrue(math.isnan(result[0]))
        self.assertEqual(result[1], float("inf"))
        self.assertEqual(result[2], 3)

    def test_inputs_not_mutated(self):
        matrix = [
            [1, 2.5],
            [-3, 4]
        ]
        vector = [0.5, 2]
        expected_matrix = copy.deepcopy(matrix)
        expected_vector = copy.deepcopy(vector)

        matrix_vector_product(matrix, vector)

        self.assertEqual(matrix, expected_matrix)
        self.assertEqual(vector, expected_vector)


if __name__ == "__main__":
    unittest.main()

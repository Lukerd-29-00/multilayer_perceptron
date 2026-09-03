#include <stddef.h>


#ifndef LINAL_TYPES
#define LINAL_TYPES

/**
 * @brief Represents a matrix
 * 
 * Matrix is a 2-d array of size height x width.
 * There is a single array for the values; use RETRIEVE to calculate the index of an item.
 * The values array is restricted.
*/
typedef struct mat {
    size_t width;
    size_t height;
    double * restrict values;
} Matrix;

#endif

/**
 * @brief adds two vectors
 * 
 * Adds two vectors together by summing their components.
 * This function is destructive; a is overwritten with a + b.
 * The inputs are restricted pointers and must point to distinct, non-overlapping arrays.
 * 
 * @param a One of the addends. Will be overwritten with the result.
 * @param b the other operand. Is not overwritten.
 * @param dimensions The dimensions of both vectors. Both vectors must have the same size or adding them is undefined.
 */
void add(double * const restrict a, const double * const restrict b, const size_t dimensions);

/**
 * @brief transforms vector x using A as a linear transformation.
 * 
 * @param A The linear transformation to apply.
 * @param x The vector to transform. Will be overwritten with A * x.
*/
void transform(const Matrix * const A, const double * restrict const x, double * restrict output);

/**
 * @brief initialize a matrix given its height and width.
 * 
 * Allocates the memory for the struct and the underlying values on the heap.
 * Must be deleted later via destroy_matrix.
 * Will return NULL if allocation fails. If the value is null check errno for the error.
 *
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @return A pointer to the matrix, or NULL if an error occurs during allocation.
 */
Matrix *initialize_matrix(const int height, const int width);

/**
 * @brief destroy a matrix created by initialize_matrix
 *
 * Frees the underlying values array along with the matrix struct itself on the heap.
 *
 * @param mat The matrix to destroy
*/
void destroy_matrix(Matrix * const mat);

/** 
 * @brief add the elements of two matrices
 * 
 * Sets the values in a to that item plus the corresponding item in b.
 * Requires both to be the same size.
 *
 * @param a The first input. Values will be overwritten.
 * @param b The second input. Will not be overwritten.
 */
void add_matrices(Matrix * const restrict a, const Matrix * const restrict b);
void scale_rows_destructive(Matrix * const target, const double * const scales);
void scale_rows(const Matrix * const restrict target, const double * const restrict scales, Matrix * const restrict output);
void transpose(const Matrix * const restrict input, Matrix * const restrict output);
void outer_product(const double * const restrict vector_a, size_t size_a, const double * const restrict vector_b, size_t size_b, Matrix *output_matrix);

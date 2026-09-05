#include "linal.h"
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

void add(double * const restrict a, const double * const restrict b, const size_t dimensions) {
    for (int i = 0; i < dimensions; i++) {
        a[i] += b[i];
    }
}

void transform(const Matrix * const A, const double * restrict const x, double * restrict output) {
    const size_t height = A->height;
    const size_t width = A->width;
    
    //possible optimization: divide the problem into squares of the matrix.
    //Do multiplications on the columns first to make use of pipelining, then add down the rows one pair of columns at a time for the same reason.
    //Dividing it into squares makes it so all computed values can fit into the memory cache.
    for (int i = 0; i < height; i++)
        output[i] = A->values[i * width] * x[0];
    
    for (int i = 1; i < width; i++) {
        for (int j = 0; j < height; j++) 
            output[j] += A->values[j * width + i] * x[i];
    }
}

void outer_product(const double * const restrict vector_a, size_t size_a, const double * const restrict vector_b, size_t size_b, Matrix *output_matrix) {
    assert(size_a == output_matrix->height);
    assert(size_b == output_matrix->width);
    const size_t height = output_matrix->height;
    const size_t width = output_matrix->width;
    
    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_b; j++) {
            output_matrix->values[i * width + j] = vector_a[i] * vector_b[j];
        }
    }
}
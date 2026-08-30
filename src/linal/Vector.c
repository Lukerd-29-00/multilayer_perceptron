#include "linal.h"
#include <stdlib.h>
#include <stdbool.h>

Vector *initialize_vec(int dimensions) {
    Vector *vec = malloc(sizeof(Vector));
    if (vec == NULL) {
        return NULL;
    }
    vec->dimensions = dimensions;
    double *values = malloc(sizeof(double) * dimensions);
    if (values == NULL) {
        return NULL;
    }
    vec->values = values;
    return vec;
}

void add(Vector * const restrict a, const Vector * const restrict b) {
    double * restrict a_values = a->values;
    const double * restrict b_values = b->values;
    for (int i = 0; i < a->dimensions; i++) {
        a_values[i] = a_values[i] + b_values[i];
    }
}

void transform(const Matrix * const A, const Vector * restrict const x, Vector * restrict output) {
    const double * const values = malloc(sizeof(double) * A->width * A->height);
    if (values == NULL) {
        return;
    }

    const double * const vec_values = x->values;
    const double * const mat_values = A->values;
    double * const output_values = output->values;
    const int height = A->height;
    const int width = A->width;
    for (int i = 0; i < height; i++) {
        output_values[i] = 0;
        for (int j = 0; j < width; j++) {
            output_values[i] += vec_values[j] * RETRIEVE(mat_values, j, i, width);
        }
    }
}

bool vec_equal(const Vector * const restrict A, const Vector * const restrict B) {
    const double * const restrict A_values = A->values;
    const double * const restrict B_values = B->values;
    for (int i = 0; i < A->dimensions; i++) {
        if (A_values[i] != B_values[i]) 
            return false;
    }
    return true;
}

void destroy_vector(Vector *vec) {
    free(vec->values);
    free(vec);
}
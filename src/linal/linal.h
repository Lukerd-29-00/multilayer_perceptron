#include <stddef.h>
#define RETRIEVE(values, i, j, w) (values[i * w + j])

#ifndef LINAL_TYPES
#define LINAL_TYPES

typedef struct mat {
    size_t width;
    size_t height;
    double * restrict values;
} Matrix;

typedef struct vec {
    int dimensions;
    double * restrict values;
} Vector;

#endif

void add(Vector * const restrict a, const Vector * const restrict b);

void transform(const Matrix * const A, const Vector * restrict const x, Vector * restrict output);

Vector *initialize_vec(int dimensions);

bool vec_equal(const Vector * const restrict A, const Vector * const restrict B);

Matrix *initialize_matrix(const int height, const int width);

void destroy_matrix(Matrix *mat);

void destroy_vector(Vector *vec);

char *test_transform_3x3(void);
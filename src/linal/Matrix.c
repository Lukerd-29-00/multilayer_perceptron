#include "linal.h"
#include <stdlib.h>
#include <assert.h>

Matrix *initialize_matrix(const int height, const int width) {
    Matrix * output = malloc(sizeof(Matrix));
    if (output == NULL) {
        return NULL;
    }
    double * values = malloc(sizeof(double) * height * width);
    if (values == NULL) {
        return NULL;
    }
    output->height = height;
    output->width = width;
    output->values = values;
    return output;
}

void add_matrices(Matrix * const restrict a, const Matrix * const restrict b) {
    assert(a->height == b->height);
    assert(a->width == b->width);
    for (int i = 0; i < a->height * a->width; i++) {
        a->values[i] += b->values[i];
    }
}

void transpose(const Matrix * const restrict input, Matrix * const restrict output) {
    assert(input->height == output->width);
    assert(input->width == output->height);
    for (int i = 0; i < input->height; i++) {
        for (int j = 0; j < input->width; j++) {
            output->values[i + j * output->width] = input->values[i * input->width + j];
        }
    }
}

void scale_rows_destructive(Matrix * const target, const double * const scales) {
    for (int i = 0; i < target->height; i++) {
        for (int j = 0; j < target->width; j++) {
            target->values[i * target->width + j] *= scales[i];
        }
    }
}

void scale_rows(const Matrix * const restrict target, const double * const restrict scales, Matrix * const restrict output) {
    assert(target->height == output->height);
    assert(target->width == output->width);
    for (int i = 0; i < target->height; i++) {
        for (int j = 0; j < target->width; j++) {
            output->values[i * target->width + j] = scales[i] * target->values[i * target->width + j];
        }
    }
}



void destroy_matrix(Matrix *mat) {
    free(mat->values);
    free(mat);
}


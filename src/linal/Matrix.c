#include "linal.h"
#include <stdlib.h>

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
    for (int i = 0; i < a->height * a->width; i++) {
        a->values[i] += b->values[i];
    }
}

void transpose(const Matrix * const restrict input, Matrix * const restrict output) {
    for (int i = 0; i < output->height; i++) {
        for (int j = 0; j < output->width; j++) {
            output[i + j * width] = input[i * width + j]
        }
    }
}

void scale_rows_destructive(Matrix * const target, const double * const scales) {
    for (int i = 0; i < target->height; i++) {
        for (int j = 0; j < target->width; j++) {
            target[i * target->height + j] *= scales[i];
        }
    }
}

void scale_rows(const Matrix * const restrict target, const double * const restrict scales, Matrix * const restrict output) {
    for (int i = 0; i < target->height; i++) {
        for (int j = 0; j < target->width; j++) {
            output[i * target->height + j] = scales[i] * target[i * target->height + j];
        }
    }
}



void destroy_matrix(Matrix *mat) {
    free(mat->values);
    free(mat);
}


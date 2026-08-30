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

void destroy_matrix(Matrix *mat) {
    free(mat->values);
    free(mat);
}


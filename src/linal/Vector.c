#include "linal.h"
#include <stdlib.h>
#include <stdbool.h>

void add(double * const restrict a, const double * const restrict b, const size_t dimensions) {
    for (int i = 0; i < dimensions; i++) {
        a[i] += b[i];
    }
}

void transform(const Matrix * const A, const double * restrict const x, double * restrict output) {
    const size_t height = A->height;
    const size_t width = A->width;
    
    
    for (int i = 0; i < height; i++)
        output[i] = A->values[i * width] * x[0];
    
        

    //Does column by column so that all references to x[i] happen one after another to take advantage of memory caching.
    for (int i = 1; i < width; i++) {
        for (int j = 0; j < height; j++) 
            output[j] += A->values[j * width + i] * x[i];
    }
}

#include <math.h>
#include "activation.h"
#include "../linal/linal.h"
#include <assert.h>

void softmax(double * restrict all_values, int num_values, double * restrict output) {
    double denominator = 0;
    for (int i = 0; i < num_values; i++) {
        output[i] = exp(all_values[i]);
        denominator += output[i];
    };

    for (int i = 0; i < num_values; i++) {
        output[i] /= denominator;
    }
}

void softmax_destructive(double *all_values, int num_values) {
    double denominator = 0;
    for (int i = 0; i < num_values; i++) {
        all_values[i] = exp(all_values[i]);
        denominator += all_values[i];
    };

    for (int i = 0; i < num_values; i++) {
        all_values[i] /= denominator;
    }
}

void softmax_prime(const double * const restrict softmax_outputs, Matrix * const output_matrix) {
    assert(output_matrix->height == output_matrix->width);
    const size_t width = output_matrix->width;
    //possible optimization: only half the matrix needs to be computed
    //Opposite sides of the diagnonal are mirrored
    double *values = output_matrix->values;
    for (int i = 0; i < output_matrix->height; i++) {
        double Si = softmax_outputs[i];
        for (int j = 0; j < i; j++) {
            values[i * width + j] = -Si * softmax_outputs[j];
        }
        values[i * width + i] = Si * (1 - Si);
        for (int j = i + 1; j < output_matrix->width; j++) {
            values[i * width + j] = -Si * softmax_outputs[j];
        }
    }
}
#include <math.h>
#include "activation.h"

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

/** Note: calculates dz/dw for z as the output for the idxth number, w as the idxth number.
* To find dx/dw for some other element of the vector x, just flip the sign.
*/
double softmax_prime(double *input_values, int idx, int num_values) {
    double denom_constant = 0;
    for (int i = 0; i < num_values; i++) {
        if (i == idx) continue;
        denom_constant += exp(input_values[i]);
    }

    double ex = exp(input_values[idx]);
    double denom = denom_constant + ex;
    denom *= denom;
    return (denom_constant * ex) / (denom);
}
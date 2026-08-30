#include <math.h>
#include "activation.h"

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_prime(double x) {
    double ex = exp(-x);
    double denominator = 1 + ex;
    denominator *= denominator;
    return ex / denominator;
}

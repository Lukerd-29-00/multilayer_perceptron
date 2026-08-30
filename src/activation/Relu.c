#include "activation.h"

double relu(double x) {
    if (x < 0) {
        return 0;
    } else {
        return x;
    }
}

double relu_prime(double x) {
    if (x < 0) {
        return 0;
    } else if (x == 0) {
        return 0.5;
    } else {
        return 1;
    }
}
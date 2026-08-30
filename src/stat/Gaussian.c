#include <stdlib.h>
#include <math.h>
#include "gaussian.h"
#define PI 3.14159265358979323846

const double root_2 = sqrt(2);

double uniform(void) {
    double r = rand();
    return r / RAND_MAX;
}

double erfinv(double x) {
    // Domain check
    if (x <= -1.0 || x >= 1.0) {
        return (x == 1.0) ? INFINITY : ((x == -1.0) ? -INFINITY : NAN);
    }
    
    double sgn = (x < 0) ? -1.0 : 1.0;
    x = (1.0 - x) * (1.0 + x); // equivalent to 1 - x^2
    
    double lnx = log(x);
    double tt1 = 2.0 / (PI * 0.147) + 0.5 * lnx;
    double tt2 = 1.0 / 0.147 * lnx;
    
    return sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2));
}

double normalize(double mean, double stddev, double p) {
    return stddev * root_2 * erfinv(2*p - 1) + mean;
}

double rand_gaussian(double mean, double stddev) {
    double p = uniform();
    return normalize(mean, stddev, p);
}
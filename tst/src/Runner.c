#include <errno.h>
#include <stdio.h>
#include "linal/linal_test.h"
#include "activation/activation_test.h"
#include "Testing.h"
#include "stat/test_gaussian.h"
#include <stdlib.h>
#include "multilayer_perceptron/basic/test_basic.h"

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (errno) {
        perror("Error in setvbuf!");
        return errno;
    }
    const int total = 26;
    int failures = 0;
    load_resources_dir();
    if (errno) {
        perror("Error loading resources!");
        return errno;
    }
    test_linal(&failures);
    if (errno) {
        perror("Panic on linear algebra!");
        return 1;
    }
    test_activation(&failures);
    if (errno) {
        perror("Panic on activation functions!");
        return 1;
    }

    gaussian_suite(&failures);
    if (errno) {
        perror("Panic on Gaussian functions!");
        return 1;
    }

    basic_perceptron_suite(&failures);
     if (errno) {
        perror("Panic on perceptron basic functions!");
        return 1;
    }

    printf("Ran %d tests: %d failed.\n", total, failures);

    return 0;
}


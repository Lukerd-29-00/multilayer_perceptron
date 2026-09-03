#include "test_gaussian.h"
#include "../Testing.h"
#include <stdio.h>
#include <stdlib.h>

void gaussian_suite(int *failures) {
    char *error_message = NULL;
    RUN_TEST(test_normalize);
    RUN_TEST(test_gaussian);
    RUN_TEST(test_erfinv);
    RUN_TEST(test_uniform);
}
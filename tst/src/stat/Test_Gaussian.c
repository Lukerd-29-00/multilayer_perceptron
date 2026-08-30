#include "..\Testing.h"
#include "test_gaussian.h"
#include "..\..\..\src\stat\gaussian.h"
#include <math.h>

char *test_normalize(void) {
    char *error_message = NULL;
    ASSERT_DOUBLE(0, normalize(0, 1, 0.5), 0.05, "normalize");
    return NULL;
}

char *test_uniform(void) {
    char *error_message = NULL;
    //Just test for a reasonable output; distribution test is validated visually in Gen_Distribution via a histogram
    double r = uniform();
    ASSERT_DOUBLE_GE(0, r, "uniform");
    ASSERT_DOUBLE_LE(1, r, "uniform");
}

char *test_gaussian(void) {
    char *error_message = NULL;
    double r = rand_gaussian(0, 1);
    //Just test for a reasonable output; distribution test is validated visually in Gen_Distribution via a histogram
    ASSERT_DOUBLE_GE(-6, r, "rand_gaussian");
    ASSERT_DOUBLE_LE(6, r, "rand_gaussian");
    return NULL;
}

char *test_erfinv(void) {
    char *error_message = NULL;
    ASSERT_DOUBLE(0.666, erfinv(0.653738869), 0.01, "erfinv");
    ASSERT_DOUBLE(-0.15, erfinv(-0.16799597), 0.01, "erfinv");
    ASSERT_NAN(erfinv(-1.01), "erfinv");
    ASSERT_NAN(erfinv(1.01), "erfinv");
    return NULL;
}
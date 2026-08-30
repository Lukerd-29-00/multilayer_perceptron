#include "../Testing.h"
#include "..\..\..\src\activation\activation.h"
#include <stdlib.h>
#include <stdio.h>

#define ERROR_MARGIN 0.001

char *test_relu_positive(void) {
    double output_1 = relu(1);
    double output_2 = relu(3);
    double output_3 = relu(0.000001);

    char * error_message = NULL;

    ASSERT_DOUBLE(1, output_1, ERROR_MARGIN, "relu");
    ASSERT_DOUBLE(3, output_2, ERROR_MARGIN, "relu");
    ASSERT_DOUBLE(0.000001, output_3, ERROR_MARGIN, "relu");

    return NULL;
}

char *test_relu_negative(void) {
    double output_1 = relu(-1);
    double output_2 = relu(-3);
    double output_3 = relu(-0.000001);

    char *error_message = NULL;

    ASSERT_DOUBLE(0, output_1, 0.001, "relu");
    ASSERT_DOUBLE(0, output_2, 0.001, "relu");
    ASSERT_DOUBLE(0, output_3, 0.001, "relu");

    return NULL;
}

char *test_relu_zero(void) {
    char buf[MAX_LEN];
    double output = relu(0);
    char * error_message = assert_double_approx(0, output, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing positive relu: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    return NULL;
}

char *test_relu_prime_positive(void) {
    char buf[MAX_LEN];
    double output_1 = relu_prime(1);
    double output_2 = relu_prime(3);
    double output_3 = relu_prime(0.000001);

    char * error_message = assert_double_approx(1, output_1, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing positive relu prime: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    error_message = assert_double_approx(1, output_2, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing positive relu prime: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    error_message = assert_double_approx(1, output_3, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing positive relu prime: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    return NULL;
}

char *test_relu_prime_negative(void) {
    char buf[MAX_LEN];
    double output_1 = relu_prime(-1);
    double output_2 = relu_prime(-3);
    double output_3 = relu_prime(-0.000001);

    char * error_message = assert_double_approx(0, output_1, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing relu prime on a negative: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    error_message = assert_double_approx(0, output_2, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing relu prime on a negative: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    error_message = assert_double_approx(0, output_3, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing relu prime on a negative: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    return NULL;
}

char *test_relu_prime_zero(void) {
    char buf[MAX_LEN];
    double output = relu_prime(0);
    char * error_message = assert_double_approx(0.5, output, 0.001);
    if (error_message != NULL) {
        sprintf(buf, "Error testing relu prime on zero: %s", error_message);
        free(error_message);
        return build_error_message(buf);
    }
    return NULL;
}
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "Testing.h"
#include "../../src/linal/linal.h"
#include "../../src/multilayer_perceptron/basic/basic.h"

char *enrich_error_message(char * restrict const error_message, const char * restrict const method_name) {
    char buf[MAX_LEN];
    sprintf(buf, "Error testing %s: %s\n", method_name, error_message);
    free(error_message);
    return build_error_message(buf);
}

char *build_error_message(const char * const error) {
    const int output_len = strlen(error);
    char * output = malloc(sizeof(char) * (output_len + 1));
    if (output == NULL) {
        return NULL;
    }
    memcpy(output, error, sizeof(char) * (output_len + 1));
    return output;
}


char * assert_int_equals(const int a, const int b) {
    if (a != b) {
        char buf[MAX_LEN] = {};
        snprintf(buf, MAX_LEN * sizeof(char), "Error: %d != %d", a, b);
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_double_approx(const double expected, const double actual, const double error) {
    if (expected == 0) {
        if (fabs(actual) > error) {
            char buf[MAX_LEN] = {};
            snprintf(buf, MAX_LEN * sizeof(char), "Error: expected |actual| <= %.2f", error);
            return build_error_message(buf);
        }
    }
    double actual_error = fabs(expected - actual) / expected;
    if (actual_error > error) {
        char buf[MAX_LEN] = {};
        snprintf(buf, MAX_LEN * sizeof(char), "Error: got error of %.2f; expected <= %.2f", actual_error, error);
        return build_error_message(buf);
    }
    return NULL;
}

char * assert_array_equals(const void * const restrict a, const void * const restrict b, const int length, const int size) {
    char buf[MAX_LEN] = {};
    if (memcmp(a, b, length*size)) {
        memcpy(buf, "Error: Arrays are not equal!", 28 * sizeof(char));
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_activation_equals(Activation expected, Activation actual) {
     if (actual != expected) {
        char buf[MAX_LEN] = {};
        snprintf(buf, MAX_LEN * sizeof(char), "Error: %d != %d", actual, expected);
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_double_ge(const double expected, const double actual) {
    char buf[MAX_LEN] = {};
    if (actual < expected) {
        snprintf(buf, MAX_LEN, "Error: got %.2f < %.2f, expected >=", actual, expected);
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_double_le(const double expected, const double actual) {
    char buf[MAX_LEN] = {};
    if (actual > expected) {
        snprintf(buf, MAX_LEN, "Error: got %.2f > %.2f, expected <=", actual, expected);
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_nan(const double actual) {
    char buf[MAX_LEN];
    if (!isnan(actual)) {
        snprintf(buf, MAX_LEN, "Error: got %.2f instead of NaN", actual);
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_null(void * actual) {
    if (actual != NULL) {
        char buf[MAX_LEN];
        snprintf(buf, MAX_LEN, "Error: expected null pointer");
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_not_null(void *actual) {
    if (actual == NULL) {
        char buf[MAX_LEN];
        snprintf(buf, MAX_LEN, "Error: got unexpected null pointer");
        return build_error_message(buf);
    }
    return NULL;
}

char *assert_pointer_equals(void *expected, void *actual) {
    if (actual != expected) {
        char buf[MAX_LEN];
        snprintf(buf, MAX_LEN, "Error: expected matching pointers!");
        return build_error_message(buf);
    }
    return NULL;
}

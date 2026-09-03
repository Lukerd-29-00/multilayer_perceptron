#include "../../../src/linal/linal.h"
#include "../Testing.h"
#include "linal_test.h"
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

char * test_matrix_init(void) {
    Matrix * const test_matrix = initialize_matrix(3, 10);
    char * error_format = NULL;

    char * error_message = assert_int_equals(test_matrix->height, 3);
    if (error_message != NULL) {
        error_format = "Error in test_matrix_init; height mismatch: %s\n";
        goto end;
    } else if (errno) {
        goto end;
    }

    error_message = assert_int_equals(test_matrix->width, 10);
    if (error_message != NULL) {
        error_format = "Error in test_matrix_init; width mismatch: %s\n";
        goto end;
    } else if (errno) {
        goto end;
    }

    end:
        destroy_matrix(test_matrix);
        if (error_format != NULL) {
            char buf[MAX_LEN] = {};
            snprintf(buf, MAX_LEN, error_format, error_message);
            char *rich_message = build_error_message(buf);
            free(error_message);
            return rich_message;
        }
        return NULL;
}

char *test_add_matrices(void) {
    char *error_message = NULL;
    Matrix *a = initialize_matrix(2, 3);
    Matrix *b = initialize_matrix(2, 3);

    a->values[0] = 1;
    a->values[1] = -1.;
    a->values[2] = 2.;
    a->values[3] = -2;
    a->values[4] = 3.5;
    a->values[5] = -3.5;

    b->values[0] = -0.9;
    b->values[1] = 1.1;
    b->values[2] = -1.9;
    b->values[3] = 2.1;
    b->values[4] = -3.4;
    b->values[5] = 3.6;

    add_matrices(a, b);
    for (int i = 0; i < 6; i++) {
        ASSERT_DOUBLE(0.1, a->values[i], 0.01, "test add matrices");
    }

    destroy_matrix(a);
    destroy_matrix(b);

    return NULL;
}

char *test_transpose(void) {
    char *error_message = NULL;
    Matrix *a = initialize_matrix(3, 2);
    Matrix *b = initialize_matrix(2, 3);

    a->values[0] = 1;
    a->values[1] = -1.;
    a->values[2] = 2.;
    a->values[3] = -2;
    a->values[4] = 3.5;
    a->values[5] = -3.5;

    transpose(a, b);

    ASSERT_DOUBLE(1, b->values[0], 0.01, "transpose");
    ASSERT_DOUBLE(2., b->values[1], 0.01, "transpose");

    ASSERT_DOUBLE(3.5, b->values[2], 0.01, "transpose");
    ASSERT_DOUBLE(-1, b->values[3], 0.01, "transpose");

    ASSERT_DOUBLE(-2, b->values[4], 0.01, "transpose");
    ASSERT_DOUBLE(-3.5, b->values[5], 0.01, "transpose");

    destroy_matrix(a);
    destroy_matrix(b);

    return NULL;
}

char *test_scale_rows_destructive(void) {
    char *error_message = NULL;

    Matrix *a = initialize_matrix(3, 2);

    a->values[0] = 1;
    a->values[1] = -1.;
    a->values[2] = 2.;
    a->values[3] = -2;
    a->values[4] = 3.5;
    a->values[5] = -3.5;

    double v[3] = {-1, 3, 2};

    scale_rows_destructive(a, v);

    ASSERT_DOUBLE(-1, a->values[0], 0.01, "scale rows (destructive)");
    ASSERT_DOUBLE(1, a->values[1], 0.01, "scale rows (destructive)");

    ASSERT_DOUBLE(6, a->values[2], 0.01, "scale rows (destructive)");
    ASSERT_DOUBLE(-6, a->values[3], 0.01, "scale rows (destructive)");

    ASSERT_DOUBLE(7, a->values[4], 0.01, "scale rows (destructive)");
    ASSERT_DOUBLE(-7, a->values[5], 0.01, "scale rows (destructive)");

    destroy_matrix(a);
    return NULL;
}

char *test_scale_rows(void) {
    char *error_message = NULL;

    Matrix *a = initialize_matrix(2, 2);
    Matrix *b = initialize_matrix(2, 2);

    a->values[0] = 1;
    a->values[1] = -1.;
    a->values[2] = 2.;
    a->values[3] = -2;

    double v[2] = {-1, 2};

    scale_rows(a, v, b);

    ASSERT_DOUBLE(-1, b->values[0], 0.01, "test scale rows");
    ASSERT_DOUBLE(1, b->values[1], 0.01, "test scale rows");
    ASSERT_DOUBLE(4, b->values[2], 0.01, "test scale rows");
    ASSERT_DOUBLE(-4, b->values[3], 0.01, "test scale rows");

    destroy_matrix(a);
    destroy_matrix(b);

    return NULL;
}

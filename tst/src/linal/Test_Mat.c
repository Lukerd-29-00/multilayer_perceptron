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

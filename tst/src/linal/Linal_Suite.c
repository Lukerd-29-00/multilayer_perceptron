#include <stdio.h>

#include "../Testing.h"
#include "linal_test.h"
#include <stdlib.h>

void test_linal(int *failures) {
    char *error_message = NULL;
    RUN_TEST(test_matrix_init);
    RUN_TEST(test_vector_add);
    RUN_TEST(test_transform_3x3);
    RUN_TEST(test_transform_3x2);
    RUN_TEST(test_transform_2x3);
    RUN_TEST(test_outer_product);
    RUN_TEST(test_add_matrices);
    RUN_TEST(test_transpose);
    RUN_TEST(test_scale_rows_destructive);
    RUN_TEST(test_scale_rows);
}

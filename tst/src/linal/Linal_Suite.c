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
}

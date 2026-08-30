#include "test_basic.h"
#include "..\..\Testing.h"
#include <stdlib.h>

void basic_perceptron_suite(int *failures) {
    char *error_message = NULL;
    RUN_TEST(test_initialize_layer);
    RUN_TEST(test_apply_transformation);
    RUN_TEST(test_apply_softmax_nondestructive);
    RUN_TEST(test_apply_activation_destructive_sigmoid);
    RUN_TEST(test_apply_activation_destructive_relu);
    RUN_TEST(test_apply_activation_destructive_softmax);
    RUN_TEST(test_build_network);
    RUN_TEST(test_feed_forward);

}
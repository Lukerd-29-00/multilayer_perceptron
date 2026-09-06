#include "../../Testing.h"
#include "test_training.h"
#include <stdlib.h>
#include <stdio.h>

void training_suite(int *failures) {
    char *error_message = NULL;
    RUN_TEST(test_glorot);
    RUN_TEST(test_he);
    RUN_TEST(test_glorot_initialization);
    RUN_TEST(test_he_initialization);
    RUN_TEST(test_initialize_for_training);

    RUN_TEST(test_calculate_backprop_for_run);
    RUN_TEST(test_calculate_dc_dinput_hidden);
    RUN_TEST(test_calculate_weight_derivs);
    RUN_TEST(test_derivative_of_sigmoid);
    RUN_TEST(test_derivative_of_relu);
    RUN_TEST(test_calculate_sigmoid_primes);
    RUN_TEST(test_calculate_relu_primes);
    RUN_TEST(test_calculate_softmax_prime);
    RUN_TEST(test_update_network);
}
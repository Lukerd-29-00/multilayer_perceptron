#include "activation_test.h"
#include "../Testing.h"
#include <stdlib.h>

void test_activation(int *failures) {
    char *error_message = NULL;

    RUN_TEST(test_softmax_prime);
    RUN_TEST(test_softmax);
    RUN_TEST(test_softmax_destructive);
    RUN_TEST(test_relu_positive);
    RUN_TEST(test_relu_negative);
    RUN_TEST(test_relu_zero);
    RUN_TEST(test_relu_prime_positive);
    RUN_TEST(test_relu_prime_negative);
    RUN_TEST(test_relu_prime_zero);
    RUN_TEST(test_sigmoid);
    RUN_TEST(test_sigmoid_prime);
}
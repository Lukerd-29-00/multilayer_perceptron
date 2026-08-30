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
    
}
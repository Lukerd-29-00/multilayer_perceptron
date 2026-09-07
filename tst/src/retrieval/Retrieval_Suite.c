#include "test_retrieval.h"
#include "../Testing.h"
#include <stdlib.h>

void test_retrieval(int *failures) {
    char *error_message = NULL;
    RUN_TEST(test_learnset_size);
    RUN_TEST(test_load);

}
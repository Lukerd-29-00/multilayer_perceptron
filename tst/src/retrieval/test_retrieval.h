#include <stddef.h>
#include "../../../src/retrieval/retrieval.h"

char *test_learnset_size();
void test_retrieval(int *failures);
void setup();
void teardown();
size_t add_training_run(Training_Run_t training_run, size_t input_values_size, size_t output_values_size);
char *test_load();
void setup_network_retrieval_test();
void teardown_network_retrieval_test();
char *test_save_and_read_network();
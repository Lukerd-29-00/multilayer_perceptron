#include <pthread.h>
#include "../multilayer_perceptron/training/training.h"
#include <sqlite3.h>

#ifndef RETRIEVAL_TYPES
#define RETRIEVAL_TYPES

#define FAILURE -1
#define SUCCESS 0
#define NO_ROWS_FOUND 1001
#define EXCESS_ROWS_FOUND 1002

typedef struct queue {
    Training_Run_t **buffer;
    int start_idx;
    int end_idx;
    size_t size;
    pthread_cond_t *cond;
    pthread_mutex_t *mutex;
} Queue;

#endif

void initialize_retrieval(sqlite3 *input_conn);
void stop_retrieval();
size_t learnset_size(const char *learnset);
void load(const char *learnset, const size_t sample_idx, const size_t sample_size, Training_Run_t *training_runs);
void load_training_run_data(int id, Training_Run_t *training_run);
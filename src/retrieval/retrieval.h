#include "../multilayer_perceptron/training/training.h"
#include <sqlite3.h>

#ifndef RETRIEVAL_TYPES
#define RETRIEVAL_TYPES

#define FAILURE -1
#define SUCCESS 0
#define NO_ROWS_FOUND 1001
#define EXCESS_ROWS_FOUND 1002


#endif

void initialize_retrieval(sqlite3 *input_conn);
void stop_retrieval();
size_t learnset_size(const char *learnset);
void load(const char *learnset, const size_t sample_idx, const size_t sample_size, Training_Run_t *training_runs);
void load_training_run_data(int id, Training_Run_t *training_run);
const char *activation_to_string(Activation activation);
Activation decode_activation(const char *name);
void initialize_transaction_stmts();
void initialize_insert_stmts();
void initialize_network_retrieval(sqlite3 *new_conn);
void destroy_network_retrieval();
void save_network(Network *network, const char * name);
void save_biases(const double *biases, int layer_id, size_t num_biases);
void save_weights(const double *weights, int layer_id, size_t num_biases);
void read_layers(const char *name, const int num_layers, Layer_Info_t *layer_infos, int *first_size, int *ids);
void read_doubles(sqlite3_stmt *stmt, const int id, const int count, double *values);
void read_values(Network *network, int *ids);
Network *read_network(const char *name);
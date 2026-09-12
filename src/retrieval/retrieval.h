#include "../multilayer_perceptron/training/training.h"
#include "../multilayer_perceptron/basic/basic.h"
#include <sqlite3.h>

#ifndef RETRIEVAL_TYPES
#define RETRIEVAL_TYPES

#define FAILURE -1
#define SUCCESS 0
#define NOT_ENOUGH_ROWS 1003
#define NO_ROWS_FOUND 1001
#define EXCESS_ROWS_FOUND 1002


#endif

void initialize_retrieval(sqlite3 *input_conn);
void stop_retrieval();
size_t learnset_size(const char *learnset);
void load(const char *learnset, const size_t sample_idx, const size_t sample_size, Training_Run_t *training_runs);
void load_training_run_data(int id, Training_Run_t *training_run);
char *activation_to_string(Activation activation);
Activation decode_activation(const char *name);
void initialize_transaction_stmts();
void initialize_insert_stmts();
void initialize_network_retrieval(sqlite3 *new_conn);
void destroy_network_retrieval();
void save_network(Network *network, const char * name);
void save_biases(const double *biases, int layer_id, size_t num_biases);
void save_weights(const double *weights, int layer_id, size_t num_biases);
void begin_transaction();
int insert_network_row(const char *);
int insert_layer_row(const int, const Layer layer, const int idx);
void add_layer(const int, const Layer layer, const int idx);
void commit();
void rollback();
void destroy_insert_stmts();
void destroy_transaction_stmts();

int get_num_layers(const char *name);
void get_layer_data(const char *name, const int layers_count, Layer_Info_t *layer_infos, int *layer_ids);
void populate_doubles(sqlite3_stmt *stmt, int num_values, double *values, const int layer_id);
void populate_layer(Layer *layer, int layer_id);
void populate_weights_biases(Network *network, const int *layer_ids);
Network *read_network(const char * name) ;
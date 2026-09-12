#include <sqlite3.h>
#include <errno.h>
#include "../multilayer_perceptron/basic/basic.h"
#include "retrieval.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define SIGMOID_NAME "SIGMOID"
#define RELU_NAME "RELU"
#define SOFTMAX_NAME "SOFTMAX"
#define NONE_NAME "NONE"

const char *create_network_tables = "CREATE TABLE IF NOT EXISTS NETWORKS (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE); CREATE INDEX IF NOT EXISTS idx_network_name ON NETWORKS(name); CREATE TABLE IF NOT EXISTS LAYERS (id INTEGER PRIMARY KEY, network_id INTEGER NOT NULL, idx INTEGER NOT NULL, size INTEGER NOT NULL, func TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_layer ON LAYERS(network_id, idx); CREATE TABLE IF NOT EXISTS BIAS_VALUES (layer_id INTEGER NOT NULL, idx INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY(layer_id, idx)); CREATE TABLE IF NOT EXISTS WEIGHT_VALUES (layer_id INTEGER NOT NULL, idx INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY(layer_id, idx))";
const char *begin_transaction_raw = "BEGIN TRANSACTION";

const char *insert_network_row_raw = "INSERT INTO NETWORKS (name) VALUES (?1)";
const char *insert_layer_row_raw = "INSERT INTO LAYERS (network_id, idx, size, func) VALUES (?1, ?2, ?3, ?4)";
const char *insert_bias_value_raw = "INSERT INTO BIAS_VALUES (layer_id, idx, value) VALUES (?1, ?2, ?3)";
const char *insert_weight_value_raw = "INSERT INTO WEIGHT_VALUES (layer_id, idx, value) VALUES (?1, ?2, ?3)";

const char *select_layers_count_raw = "SELECT COUNT(*) FROM NETWORKS INNER JOIN LAYERS ON NETWORKS.id = LAYERS.network_id WHERE name = ?1";
const char *select_layer_rows_raw = "SELECT size, func, LAYERS.id FROM NETWORKS INNER JOIN LAYERS ON NETWORKS.id = LAYERS.network_id WHERE name = ?1 ORDER BY idx";
const char *select_bias_values_raw = "SELECT value FROM BIAS_VALUES WHERE layer_id = ?1 ORDER BY idx";
const char *select_weight_values_raw = "SELECT value FROM WEIGHT_VALUES WHERE layer_id = ?1 ORDER BY idx";

const char *commit_raw = "COMMIT";
const char *rollback_raw = "ROLLBACK";

sqlite3_stmt *begin_transaction_stmt;

sqlite3_stmt *insert_network_row_stmt;
sqlite3_stmt *insert_layer_row_stmt;
sqlite3_stmt *insert_bias_value_stmt;
sqlite3_stmt *insert_weight_value_stmt;

sqlite3_stmt *select_layers_count_stmt;
sqlite3_stmt *select_layer_rows_stmt;
sqlite3_stmt *select_bias_values_stmt;
sqlite3_stmt *select_weight_values_stmt;

sqlite3_stmt *commit_stmt;
sqlite3_stmt *rollback_stmt;

sqlite3 *network_retrieval_conn;

Activation decode_activation(const char *activation_name) {
    if (!strcmp(SIGMOID_NAME, activation_name)) {
        return SIGMOID;
    } else if (!strcmp(RELU_NAME, activation_name)) {
        return RELU;
    } else if (!strcmp(SOFTMAX_NAME, activation_name)) {
        return SOFTMAX;
    } else if (!strcmp(NONE_NAME, activation_name)) {
        return NONE;
    }

    assert(1!=1);
    return NONE;
}

char *activation_to_string(Activation activation) {
    switch (activation) {
        case SIGMOID:
            return SIGMOID_NAME;
        case RELU:
            return RELU_NAME;
        case SOFTMAX:
            return SOFTMAX_NAME;
        case NONE:
            return NONE_NAME;
    }
}

void initialize_transaction_stmts() {
    int ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        begin_transaction_raw,
        1024,
        &begin_transaction_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        commit_raw,
        1024,
        &commit_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(begin_transaction_stmt);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        rollback_raw,
        1024,
        &rollback_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(begin_transaction_stmt);
        sqlite3_finalize(commit_stmt);
        return;
    }
}

void initialize_insert_stmts() {
    int ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        insert_network_row_raw,
        1024,
        &insert_network_row_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        insert_layer_row_raw,
        1024,
        &insert_layer_row_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(insert_network_row_stmt);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        insert_bias_value_raw,
        1024,
        &insert_bias_value_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(insert_network_row_stmt);
        sqlite3_finalize(insert_layer_row_stmt);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        insert_weight_value_raw,
        1024,
        &insert_weight_value_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(insert_network_row_stmt);
        sqlite3_finalize(insert_layer_row_stmt);
        sqlite3_finalize(insert_bias_value_stmt);
        return;
    }
}

void initialize_select_stmts() {
    int ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        select_layers_count_raw,
        1024,
        &select_layers_count_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        select_layer_rows_raw,
        1024,
        &select_layer_rows_stmt,
        NULL
    );

    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(select_layers_count_stmt);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        select_bias_values_raw,
        1024,
        &select_bias_values_stmt,
        NULL
    );
    if (ok != SQLITE_OK)  {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(select_layers_count_stmt);
        sqlite3_finalize(select_layer_rows_stmt);
        return;
    }

    ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        select_weight_values_raw,
        1024,
        &select_weight_values_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        sqlite3_finalize(select_layers_count_stmt);
        sqlite3_finalize(select_layer_rows_stmt);
        sqlite3_finalize(select_bias_values_stmt);
    }
}

void initialize_network_retrieval(sqlite3 *new_network_retrieval_conn) {
    network_retrieval_conn = new_network_retrieval_conn;

    int ok = sqlite3_exec(
        network_retrieval_conn,
        create_network_tables,
        NULL,
        NULL,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(network_retrieval_conn);
        return;
    }

    initialize_transaction_stmts();
    if (errno) {
        return;
    }
    initialize_insert_stmts();
    if (errno) {
        destroy_transaction_stmts();
        return;
    }
    initialize_select_stmts();
    if (errno) {
        destroy_transaction_stmts();
        destroy_insert_stmts();
    }
}

void destroy_transaction_stmts() {
    sqlite3_finalize(begin_transaction_stmt);
    sqlite3_finalize(rollback_stmt);
    sqlite3_finalize(commit_stmt);
}

void destroy_insert_stmts() {
    sqlite3_finalize(insert_network_row_stmt);
    sqlite3_finalize(insert_layer_row_stmt);
    sqlite3_finalize(insert_bias_value_stmt);
    sqlite3_finalize(insert_weight_value_stmt);
}

void destroy_select_stmts() {
    sqlite3_finalize(select_layers_count_stmt);
    sqlite3_finalize(select_layer_rows_stmt);
    sqlite3_finalize(select_bias_values_stmt);
    sqlite3_finalize(select_weight_values_stmt);
}

void destroy_network_retrieval() {
    destroy_transaction_stmts();
    destroy_insert_stmts();
    destroy_select_stmts();
}

void begin_transaction() {
    int ok = sqlite3_step(begin_transaction_stmt);
    sqlite3_reset(begin_transaction_stmt);
    if (ok != SQLITE_DONE) {
        errno = sqlite3_errcode(network_retrieval_conn);
    }
}

int insert_network_row(const char * name) {
    sqlite3_bind_text(insert_network_row_stmt, 1, name, strlen(name), SQLITE_STATIC);
    int ok = sqlite3_step(insert_network_row_stmt);
    int network_id = FAILURE;
    if (ok == SQLITE_DONE) {
        network_id = sqlite3_last_insert_rowid(network_retrieval_conn);
    } else {
        errno = sqlite3_errcode(network_retrieval_conn);
    }
    sqlite3_reset(insert_network_row_stmt);
    sqlite3_clear_bindings(insert_network_row_stmt);
    return network_id;
}

int insert_layer_row(const int network_id, const Layer layer, const int idx) {
    int layer_id = FAILURE;
    sqlite3_bind_int(insert_layer_row_stmt, 1, network_id);
    sqlite3_bind_int(insert_layer_row_stmt, 2, idx);
    sqlite3_bind_int(insert_layer_row_stmt, 3, layer.size);
    const char *activation_name = activation_to_string(layer.func);
    sqlite3_bind_text(insert_layer_row_stmt, 4, activation_name, strlen(activation_name), SQLITE_STATIC);
    int ok = sqlite3_step(insert_layer_row_stmt);
    if (ok == SQLITE_DONE) {
        layer_id = sqlite3_last_insert_rowid(network_retrieval_conn);
    } else {
        errno = sqlite3_errcode(network_retrieval_conn);
    }
    sqlite3_reset(insert_layer_row_stmt);
    sqlite3_clear_bindings(insert_layer_row_stmt);
    return layer_id;
}

void add_layer(const int network_id, const Layer layer, const int idx) {
    int layer_id = insert_layer_row(network_id, layer, idx);
    if (errno) {
        return;
    }
    assert((layer.biases != NULL && layer.incoming_weights.values != NULL) || (layer.biases == NULL && layer.incoming_weights.values == NULL));
    if (layer.biases != NULL && layer.incoming_weights.values != NULL) {
        save_biases(layer.biases, layer_id, layer.size);
        if (errno) {
            return;
        }
        save_weights(layer.incoming_weights.values, layer_id, layer.incoming_weights.width * layer.incoming_weights.height);
    }
}

void commit() {
    int ok = sqlite3_step(commit_stmt);
    sqlite3_reset(commit_stmt);
    if (ok != SQLITE_DONE) {
        errno = sqlite3_errcode(network_retrieval_conn);
    }

}

void rollback() {
    //No ok check; if we rollback we don't want to overwrite the existing error.
    sqlite3_step(rollback_stmt);
    sqlite3_reset(rollback_stmt);
}

void save_network(Network *network, const char * name) {
    begin_transaction();
    if (errno) {
        goto rollback_gt;
    }

    int network_id = insert_network_row(name);
    if (errno) {
        goto rollback_gt;
    }

    for (int i = 0; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        add_layer(network_id, layer, i);
        if (errno) {
            goto rollback_gt;
        }
    }

    commit();
    return;

    rollback_gt:
    rollback();
}

void save_biases(const double *biases, int layer_id, size_t num_biases) {
    for (int i = 0; i < num_biases; i++) {
        sqlite3_bind_int(insert_bias_value_stmt, 1, layer_id);
        sqlite3_bind_int(insert_bias_value_stmt, 2, i);
        sqlite3_bind_double(insert_bias_value_stmt, 3, biases[i]);
        int ok = sqlite3_step(insert_bias_value_stmt);
        sqlite3_reset(insert_bias_value_stmt);
        sqlite3_clear_bindings(insert_bias_value_stmt);
        if (ok != SQLITE_DONE) {
            errno = sqlite3_errcode(network_retrieval_conn);
            return;
        }

    }
    
}

void save_weights(const double *weights, int layer_id, size_t num_biases) {
    for (int i = 0; i < num_biases; i++) {
        sqlite3_bind_int(insert_weight_value_stmt, 1, layer_id);
        sqlite3_bind_int(insert_weight_value_stmt, 2, i);
        sqlite3_bind_double(insert_weight_value_stmt, 3, weights[i]);
        int ok = sqlite3_step(insert_weight_value_stmt);
        sqlite3_reset(insert_weight_value_stmt);
        sqlite3_clear_bindings(insert_weight_value_stmt);
        if (ok != SQLITE_DONE) {
            errno = sqlite3_errcode(network_retrieval_conn);
            return;
        }
    }
}


int get_num_layers(const char *name) {
    sqlite3_bind_text(select_layers_count_stmt, 1, name, strlen(name), SQLITE_STATIC);
    int ok = sqlite3_step(select_layers_count_stmt);
    int output = FAILURE;
    if (ok == SQLITE_ROW) {
        output = sqlite3_column_int(select_layers_count_stmt, 0);
    } else {
        errno = sqlite3_errcode(network_retrieval_conn);
    }
    sqlite3_reset(select_layers_count_stmt);
    sqlite3_clear_bindings(select_layers_count_stmt);
    return output;
}

void get_layer_data(const char *name, const int layers_count, Layer_Info_t *layer_infos, int *layer_ids) {
    sqlite3_bind_text(select_layer_rows_stmt, 1, name, strlen(name), SQLITE_STATIC);
    for (int i = 0; i < layers_count; i++) {
        int ok = sqlite3_step(select_layer_rows_stmt);
        if (ok != SQLITE_ROW) {
            errno = sqlite3_errcode(network_retrieval_conn);
            if (errno == 0) {
                errno = NOT_ENOUGH_ROWS;
            }
            break;
        }
        layer_infos[i].size = sqlite3_column_int(select_layer_rows_stmt, 0);
        layer_infos[i].activation = decode_activation(sqlite3_column_text(select_layer_rows_stmt, 1));
        layer_ids[i] = sqlite3_column_int(select_layer_rows_stmt, 2);
    }
    sqlite3_reset(select_layer_rows_stmt);
    sqlite3_clear_bindings(select_layer_rows_stmt);
}

void populate_doubles(sqlite3_stmt *stmt, int num_values, double *values, const int layer_id) {
    sqlite3_bind_int(stmt, 1, layer_id);
    for (int i = 0; i < num_values; i++) {
        int ok = sqlite3_step(stmt);
        if (ok != SQLITE_ROW) {
            errno = sqlite3_errcode(network_retrieval_conn);
            if (errno == 0) {
                errno = NOT_ENOUGH_ROWS;
            }
            break;
        }
        values[i] = sqlite3_column_double(stmt, 0);
    }
    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
}

void populate_weights_biases(Network *network, const int *layer_ids) {
    for (int i = 1; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        populate_doubles(select_bias_values_stmt, layer.size, layer.biases, layer_ids[i]);
        if (errno) {
            return;
        }
        populate_doubles(select_weight_values_stmt, layer.incoming_weights.width * layer.incoming_weights.height, layer.incoming_weights.values, layer_ids[i]);
        if (errno) {
            return;
        }
    }
}

Network *read_network(const char * name) {
    begin_transaction();
    int *layer_ids = NULL;
    Layer_Info_t *layer_info = NULL;
    Network *network = NULL;
    int num_layers = get_num_layers(name);
    if (errno) {
        goto rollback_rn;
    }
    layer_info = malloc(sizeof(Layer_Info_t) * num_layers);
    if (layer_info == NULL) {
        goto rollback_rn;
    }
    layer_ids = malloc(sizeof(int) * num_layers);
    if (layer_ids == NULL) {
        goto rollback_rn;
    }
    get_layer_data(name, num_layers, layer_info, layer_ids);
    if (errno) {
        goto rollback_rn;
    }
    network = build_network(layer_info[0].size, layer_info+1, num_layers-1);
    if (network == NULL) {
        goto rollback_rn;
    }
    populate_weights_biases(network, layer_ids);
    if (errno) {
        goto rollback_rn;
    }

    free(layer_ids);
    free(layer_info);
    commit();
    return network;

    rollback_rn:
    if (layer_ids != NULL) {
        free(layer_ids);
    }
    if (layer_info != NULL) {
        free(layer_info);
    }
    if (network != NULL) {
        destroy_network(network);
    }
    rollback();
    return NULL;
}
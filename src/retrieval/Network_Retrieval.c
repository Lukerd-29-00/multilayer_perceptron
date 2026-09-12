#include <sqlite3.h>
#include <errno.h>
#include "../multilayer_perceptron/basic/basic.h"
#include "retrieval.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const char *sigmoid_name = "SIGMOID";
const char *relu_name = "RELU";
const char *softmax_name = "SOFTMAX";
const char *none_name = "NONE";

const char *create_network_tables = "CREATE TABLE IF NOT EXISTS NETWORKS (id INTEGER PRIMARY KEY, name TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_network_name ON NETWORKS(name); CREATE TABLE IF NOT EXISTS LAYERS (id INTEGER PRIMARY KEY, network_id INTEGER NOT NULL, idx INTEGER NOT NULL, size INTEGER NOT NULL, func TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_layer ON LAYERS(network_id, idx); CREATE TABLE IF NOT EXISTS BIAS_VALUES (layer_id INTEGER NOT NULL, idx INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY(layer_id, idx)); CREATE TABLE IF NOT EXISTS WEIGHT_VALUES (layer_id INTEGER NOT NULL, idx INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY(layer_id, idx))";
const char *begin_transaction = "BEGIN TRANSACTION";

const char *insert_network_row = "INSERT INTO NETWORKS (name) VALUES (?1)";
const char *insert_layer_row = "INSERT INTO LAYERS (network_id, idx, size, func) VALUES (?1, ?2, ?3, ?4)";
const char *insert_bias_value = "INSERT INTO BIAS_VALUES (layer_id, idx, value) VALUES (?1, ?2, ?3)";
const char *insert_weight_value = "INSERT INTO WEIGHT_VALUES (layer_id, idx, value) VALUES (?1, ?2, ?3)";

const char *commit = "COMMIT";
const char *rollback = "ROLLBACK";

sqlite3_stmt *begin_transaction_stmt;

sqlite3_stmt *insert_network_row_stmt;
sqlite3_stmt *insert_layer_row_stmt;
sqlite3_stmt *insert_bias_value_stmt;
sqlite3_stmt *insert_weight_value_stmt;

sqlite3_stmt *commit_stmt;
sqlite3_stmt *rollback_stmt;

sqlite3 *network_retrieval_conn;

const char *activation_to_string(Activation activation) {
    switch (activation) {
        case SIGMOID:
            return sigmoid_name;
        case RELU:
            return relu_name;
        case SOFTMAX:
            return softmax_name;
        case NONE:
            return none_name;
    }
}

void initialize_transaction_stmts() {
    int ok = sqlite3_prepare_v2(
        network_retrieval_conn,
        begin_transaction,
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
        commit,
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
        rollback,
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
        insert_network_row,
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
        insert_layer_row,
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
        insert_bias_value,
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
        insert_weight_value,
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
}

void destroy_network_retrieval() {
    sqlite3_finalize(begin_transaction_stmt);
    sqlite3_finalize(insert_network_row_stmt);
    sqlite3_finalize(insert_layer_row_stmt);
    sqlite3_finalize(insert_bias_value_stmt);
    sqlite3_finalize(insert_weight_value_stmt);
    sqlite3_finalize(commit_stmt);
    sqlite3_finalize(rollback_stmt);
}

void save_network(Network *network, const char * name) {
    int ok = sqlite3_step(begin_transaction_stmt);
    sqlite3_reset(begin_transaction_stmt);
    if (ok != SQLITE_DONE) {
        goto rollback;
    }

    sqlite3_bind_text(insert_network_row_stmt, 1, name, strlen(name), SQLITE_STATIC);

    ok = sqlite3_step(insert_network_row_stmt);
    int network_id = sqlite3_last_insert_rowid(network_retrieval_conn);
    sqlite3_reset(insert_network_row_stmt);
    sqlite3_clear_bindings(insert_network_row_stmt);
    if (ok != SQLITE_DONE) {
        goto rollback;
    }

    for (int i = 0; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        sqlite3_bind_int(insert_layer_row_stmt, 1, network_id);
        sqlite3_bind_int(insert_layer_row_stmt, 2, i);
        sqlite3_bind_int(insert_layer_row_stmt, 3, layer.size);
        const char *activation_name = activation_to_string(layer.func);
        sqlite3_bind_text(insert_layer_row_stmt, 4, activation_name, strlen(activation_name), SQLITE_STATIC);
        ok = sqlite3_step(insert_layer_row_stmt);
        int layer_id = sqlite3_last_insert_rowid(network_retrieval_conn);
        sqlite3_reset(insert_layer_row_stmt);
        sqlite3_clear_bindings(insert_layer_row_stmt);
        if (ok != SQLITE_DONE) {
            goto rollback;
        }
        if (i != 0) {
            save_biases(layer.biases, layer_id, layer.size);
            if (errno != 0) {
                goto rollback;
            }
            save_weights(layer.incoming_weights.values, layer_id, layer.incoming_weights.width * layer.incoming_weights.height);
            if (errno != 0) {
                goto rollback;
            }
        }
        
    }

    sqlite3_step(commit_stmt);
    sqlite3_reset(commit_stmt);
    return;

    rollback:
    errno = sqlite3_errcode(network_retrieval_conn);
    sqlite3_step(rollback_stmt);
    sqlite3_reset(rollback_stmt);
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
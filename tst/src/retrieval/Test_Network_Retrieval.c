#include "test_retrieval.h"
#include "../../../src/retrieval/retrieval.h"
#include "../Testing.h"
#include "../../../src/multilayer_perceptron/basic/basic.h"
#include <sqlite3.h>
#include <errno.h>
#include <stdio.h>
sqlite3 *test_network_retrieval_conn;


void setup_network_retrieval_test() {
    int ok = sqlite3_open(":memory:", &test_network_retrieval_conn);
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(test_network_retrieval_conn);
        return;
    }
    initialize_network_retrieval(test_network_retrieval_conn);
    if (errno) {
        sqlite3_close(test_network_retrieval_conn);
    }
}

void teardown_network_retrieval_test() {
    if (errno) {
        printf("%s\n", sqlite3_errstr(errno));
    }
    destroy_network_retrieval();
    sqlite3_close(test_network_retrieval_conn);
}

char *test_save_and_read_network() {
    setup_network_retrieval_test();
    if (errno) {
        printf("%s\n", sqlite3_errstr(errno));
        teardown_network_retrieval_test();
        return NULL;
    }

    Layer_Info_t info = {
        .size = 1,
        .activation = SIGMOID
    };
    Layer_Info_t infos[1] = {info};

    Network *network = build_network(1, infos, 1);
    if (network == NULL) {
        teardown_network_retrieval_test();
        return NULL;
    }

    network->layers_array[1].incoming_weights.values[0] = 1.45;
    network->layers_array[1].biases[0] = -2.24;

    save_network(network, "test");
    if (errno) {
        goto end;
    }
   
    Network *loaded_network = read_network("test");
    if (loaded_network == NULL) {
        goto end;
    }

    char *error_message = NULL;

    ASSERT_INT_EQUALS(2, loaded_network->layers_count, "test save and read network");
    Layer first = loaded_network->layers_array[0];
    ASSERT_INT_EQUALS(1, first.size, "test save and read network");
    
    Layer second = loaded_network->layers_array[1];
    ASSERT_INT_EQUALS(1, second.size, "test save and read network");
    ASSERT_INT_EQUALS(1, second.incoming_weights.width, "test save and read network");
    ASSERT_INT_EQUALS(1, second.incoming_weights.height, "test save and read network");
    ASSERT_DOUBLE(1.45, second.incoming_weights.values[0], 0.01, "test save and read network");
    ASSERT_DOUBLE(-2.24, second.biases[0], 0.01, "test save and read network");
    ASSERT_ACTIVATION_EQUALS(SIGMOID, second.func, "test save and read network");


    destroy_network(loaded_network);
    end:
    destroy_network(network);
    teardown_network_retrieval_test();
    return NULL;

}
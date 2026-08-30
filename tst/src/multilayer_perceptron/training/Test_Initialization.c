#include "test_training.h"
#include "../../Testing.h"
#include "../../../../src\multilayer_perceptron\training\training.h"
#include "../../../../src/multilayer_perceptron/basic/basic.h"
#include <math.h>
#include <stddef.h>

char *test_glorot(void) {
    size_t fan_in = 3;
    size_t fan_out = 4;

    double gl = glorot(fan_in, fan_out);

    double stddev = sqrt(2. / (fan_in + fan_out));

    char *error_message = NULL;
    ASSERT_DOUBLE_GE((-6) * stddev, gl, "Glorot");
    ASSERT_DOUBLE_LE(6 * stddev, gl, "Glorot");

    return NULL;
}

char *test_he(void) {
    size_t fan_in = 3;
    double h = he(fan_in);

    double stddev = sqrt(2. / fan_in);

    char *error_message = NULL;
    ASSERT_DOUBLE_GE((-6) * stddev, h, "Glorot");
    ASSERT_DOUBLE_LE(6 * stddev, h, "Glorot");

    return NULL;
}

char *test_glorot_initialization(void) {
    Layer *first = initialize_first_layer(3);
    Layer *second = initialize_layer(3, SIGMOID, 3);

    double stddev = sqrt(1./3.);

    first->next = second;
    second->prev = first;
    initialize_layer_glorot(second);

    char *error_message = NULL;

    for (int i = 0; i < 3; i++) {
        ASSERT_DOUBLE_GE( -6 * stddev, second->incoming_weights.values[i], "glorot layer initialization");
        ASSERT_DOUBLE_LE( 6 * stddev, second->incoming_weights.values[i], "glorot layer initialization");
    }
    return NULL;
}

char *test_he_initialization(void) {
    Layer *first = initialize_first_layer(3);
    Layer *second = initialize_layer(3, RELU, 3);

    double stddev = sqrt(2./3.);

    first->next = second;
    second->prev = first;
    initialize_layer_he(second);

    char *error_message = NULL;

    for (int i = 0; i < 3; i++) {
        ASSERT_DOUBLE_GE( -6 * stddev, second->incoming_weights.values[i], "He layer initialization");
        ASSERT_DOUBLE_LE( 6 * stddev, second->incoming_weights.values[i], "He layer initialization");
    }
    return NULL;
}

char *test_initialize_for_training(void) {
    Layer_Info_t hidden_layer_info;
    Layer_Info_t output_layer_info;

    hidden_layer_info.size = 3;
    output_layer_info.size = 1;

    Layer_Info_t layer_infos[2] = {hidden_layer_info, output_layer_info};

    Layer *network = build_network(2, layer_infos, 2);
    if (network == NULL) {
        return NULL;
    }

    initialize_for_training(network);

    Layer *hidden_layer = network->next;
    Layer *output_layer = hidden_layer->next;

    double stddev = 0.5; //sqrt 1/4

    char *error_message = NULL;

    for (int i = 0; i < 3; i++) {
        ASSERT_DOUBLE_GE( -6 * stddev, hidden_layer->incoming_weights.values[i], "network initialization");
        ASSERT_DOUBLE_LE( 6 * stddev, hidden_layer->incoming_weights.values[i], "network initialization");
    }

    stddev = 1;
    output_layer = hidden_layer->next;

    ASSERT_DOUBLE_GE( -6 * stddev, output_layer->incoming_weights.values[0], "network initialization");
    ASSERT_DOUBLE_LE( 6 * stddev, output_layer->incoming_weights.values[0], "network initialization");
    

    destroy_network(network);
    return NULL;
}
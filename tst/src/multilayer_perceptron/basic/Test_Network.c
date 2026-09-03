#include "test_basic.h"
#include "../../../../src/multilayer_perceptron/basic/basic.h"
#include "../../../../src/linal/linal.h"
#include "../../Testing.h"
#include <stdlib.h>


char *test_build_network(void) {
    Layer_Info_t layers[2];
    
    Layer_Info_t layer_two;
    layer_two.size = 3;
    layer_two.activation = RELU;

    Layer_Info_t layer_three;
    layer_three.size = 2;
    layer_three.activation = SOFTMAX;

    char *error_message = NULL;

    layers[0] = layer_two;
    layers[1] = layer_three;

    Network *network = build_network(1, layers, 2);

    ASSERT_NOT_NULL(network, "build network");
    ASSERT_INT_EQUALS(3, network->largest_layer_size, "build network");
    ASSERT_INT_EQUALS(3, network->layers_count, "build network");

    Layer start = network->layers_array[0];

    ASSERT_INT_EQUALS(1, start.size, "build network");
    ASSERT_INT_EQUALS(NO_INCOMING_WEIGHTS, start.incoming_weights.height, "build network");
    ASSERT_INT_EQUALS(NO_INCOMING_WEIGHTS, start.incoming_weights.width, "build network");
    ASSERT_NULL(start.incoming_weights.values, "build network");
    ASSERT_NULL(start.biases, "build network");
    ASSERT_ACTIVATION_EQUALS(NONE, start.func, "build network");
    ASSERT_INT_EQUALS(NO_FAN, start.fan_in, "build network");
    ASSERT_INT_EQUALS(3, start.fan_out, "build network");

    Layer hidden = network->layers_array[1];

    ASSERT_INT_EQUALS(3, hidden.size, "build network");
    ASSERT_INT_EQUALS(3, hidden.incoming_weights.height, "build network");
    ASSERT_INT_EQUALS(1, hidden.incoming_weights.width, "build network");
    ASSERT_NOT_NULL(hidden.incoming_weights.values, "build network");
    ASSERT_NOT_NULL(hidden.biases, "build network");
    ASSERT_ACTIVATION_EQUALS(RELU, hidden.func, "build network");
    ASSERT_INT_EQUALS(1, hidden.fan_in, "build network");
    ASSERT_INT_EQUALS(2, hidden.fan_out, "build network");

    Layer last = network->layers_array[2];
    ASSERT_INT_EQUALS(2, last.size, "build network");
    ASSERT_INT_EQUALS(2, last.incoming_weights.height, "build network");
    ASSERT_INT_EQUALS(3, last.incoming_weights.width, "build network");
    ASSERT_NOT_NULL(last.incoming_weights.values, "build network");
    ASSERT_NOT_NULL(last.biases, "build network");
    ASSERT_ACTIVATION_EQUALS(SOFTMAX, last.func, "build network");
    ASSERT_INT_EQUALS(3, last.fan_in, "build network");
    ASSERT_INT_EQUALS(NO_FAN, last.fan_out, "build network");
    
    destroy_network(network);

    return NULL;
}

char *test_feed_forward(void) {

    Layer_Info_t layers[2];

    Layer_Info_t layer_two;
    layer_two.size = 3;
    layer_two.activation = RELU;

    Layer_Info_t layer_three;
    layer_three.size = 2;
    layer_three.activation = SOFTMAX;

    layers[0] = layer_two;
    layers[1] = layer_three;

    Network *network = build_network(1, layers, 2);

    double input_vec[1];
    input_vec[0] = 1;

    Layer hidden = network->layers_array[1];
    double * hidden_weights = hidden.incoming_weights.values;
    hidden_weights[0] = 3;
    hidden_weights[1] = -10;
    hidden_weights[2] = 0.5;

    double *hidden_biases = hidden.biases;
    hidden_biases[0] = 0.1;
    hidden_biases[1] = 1;
    hidden_biases[2] = 3.14;

    Layer last = network->layers_array[2];

    double *last_weights = last.incoming_weights.values;
    last_weights[0] = 1.5;
    last_weights[1] = 1;
    last_weights[2] = 4.5;
    last_weights[3] = -2.33;
    last_weights[4] = -3;
    last_weights[5] = 4.5;

    last.biases[0] = -2.5;
    last.biases[1] = 3;

    double state_vec_1[3];
    double state_vec_2[3];

    double *output_vec = feed_forward(network, input_vec, state_vec_1, state_vec_2);

    char *error_message = NULL;

    ASSERT_DOUBLE(0.9982958785423037, output_vec[0], 0.01, "feed forward");
    ASSERT_DOUBLE(0.0017041214576962926, output_vec[1], 0.01, "feed forward");

    destroy_network(network);

    return NULL;
}

#include "test_training.h"
#include "../../Testing.h"
#include "../../../../multilayer_perceptron/training/training.h"
#include "../../../../linal/linal.h"
#include "../../../../multilayer_perceptron/basic/basic.h"

char *test_update_network(void) {
    char *error_message = NULL;

    Layer_Info_t layers[2];

    layers[0] = {.size = 2, .activation = SIGMOID};
    layers[1] = {.size = 3, .activation = SOFTMAX};

    Network *network = build_network(3, layers, 2);
    if (network == NULL) {
        return NULL;
    }

    double *biases_1 = network->layers_array[1].biases;
    double *biases_2 = network->layers_array[2].biases;

    biases_1.biases[0] = 1;
    biases_1.biases[1] = 2;

    biases_2.biases[0] = -1;
    biases_2.biases[1] = -1.5;
    biases_2.biases[2] = -2.3;

    double * weights_1 = network->layers_array[1].incoming_weights.values;

    for (int i = 0; i < 6; i++) {
        weights_1[i] = 1<<i;
    }

    double *weights_2 = network->layers_array[2].incoming_weights.values;

    for (int i = 0; i < 6; i++) {
        weights_2[i] = 6 - i;
    }

    Backprop_Output_t prop_outputs[2];

    prop_outputs[0] = {.weight_derivs = initialize_matrix(2, 3), .bias_derivs = {2, 3}};
    prop_outputs[1] = {.weight_derivs = initialize_matrix(3, 2), .bias_derivs = {1.4, 2, 3}};

    for (int i = 0; i < 6; i++) {
        prop_outputs[0].weight_derivs[i] = 1<<i;
        prop_outputs[1].weight_derivs[i] = 6 - i;
    }
    
    update_network(network, prop_outputs);

    ASSERT_DOUBLE(2, weights_1[0], 0.01, "test update network");
    ASSERT_DOUBLE(4, weights_1[1], 0.01, "test update network");
    ASSERT_DOUBLE(8, weights_1[2], 0.01, "test update network");
    ASSERT_DOUBLE(16, weights_1[3], 0.01, "test update network");
    ASSERT_DOUBLE(32, weights_1[4], 0.01, "test update network");
    ASSERT_DOUBLE(64, weights_1[5], 0.01, "test update newtork");

    ASSERT_DOUBLE(3, biases_1[0], 0.01, "test update network");
    ASSERT_DOUBLE(5, biases_1[1], 0.01, "test update network");

    ASSERT_DOUBLE(12, weights_2[0], 0.01, "test update network");
    ASSERT_DOUBLE(10, weights_2[1], 0.01, "test update network");
    ASSERT_DOUBLE(8, weights_2[2], 0.01, "test update network");
    ASSERT_DOUBLE(6, weights_2[3], 0.01, "test update network");
    ASSERT_DOUBLE(4, weights_2[4], 0.01, "test update network");
    ASSERT_DOUBLE(2, weights_2[5], 0.01, "test update network");

    ASSERT_DOUBLE(0.4, biases_2[0], 0.01, "test update network");
    ASSERT_DOUBLE(0.5, biases_2[1], 0.01, "test update network");
    ASSERT_DOUBLE(0.7, biases_2[2], 0.01, "test update network");

    destroy_network(network);
    return NULL;
}
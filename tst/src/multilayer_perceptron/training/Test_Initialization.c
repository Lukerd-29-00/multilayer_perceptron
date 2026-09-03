#include "test_training.h"
#include "../../Testing.h"
#include "../../../../src\multilayer_perceptron/training/training.h"
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
    Layer layer = initialize_layer(3, SIGMOID, 3, 2);

    double stddev = sqrt(2./5.);

    
    initialize_layer_glorot(layer);

    char *error_message = NULL;

    for (int i = 0; i < 3; i++) {
        ASSERT_DOUBLE_GE( -6 * stddev, layer.incoming_weights.values[i], "glorot layer initialization");
        ASSERT_DOUBLE_LE( 6 * stddev, layer.incoming_weights.values[i], "glorot layer initialization");
    }
    return NULL;
}

char *test_he_initialization(void) {
    
    Layer layer = initialize_layer(3, RELU, 3, 3);

    double stddev = sqrt(2./3.);

    initialize_layer_he(layer);

    char *error_message = NULL;

    for (int i = 0; i < 3; i++) {
        ASSERT_DOUBLE_GE( -6 * stddev, layer.incoming_weights.values[i], "He layer initialization");
        ASSERT_DOUBLE_LE( 6 * stddev, layer.incoming_weights.values[i], "He layer initialization");
    }
    return NULL;
}

char *test_initialize_for_training(void) {
    Layer_Info_t hidden_layer_info;
    Layer_Info_t output_layer_info;

    hidden_layer_info.size = 3;
    output_layer_info.size = 1;

    Layer_Info_t layer_infos[2] = {hidden_layer_info, output_layer_info};

    Network *network = build_network(2, layer_infos, 2);
    if (network == NULL) {
        return NULL;
    }

    Layer hidden_layer = network->layers_array[1];

    initialize_for_training(network);
    double stddev = sqrt(1./3.);

    char *error_message = NULL;

    for (int i = 0; i < 3; i++) {
        ASSERT_DOUBLE_GE( -6 * stddev, hidden_layer.incoming_weights.values[i], "network initialization");
        ASSERT_DOUBLE_LE( 6 * stddev, hidden_layer.incoming_weights.values[i], "network initialization");
    }

    stddev = 1;
    Layer output_layer = network->layers_array[2];

    ASSERT_DOUBLE_GE( -6 * stddev, output_layer.incoming_weights.values[0], "network initialization");
    ASSERT_DOUBLE_LE( 6 * stddev, output_layer.incoming_weights.values[0], "network initialization");
    

    destroy_network(network);
    return NULL;
}
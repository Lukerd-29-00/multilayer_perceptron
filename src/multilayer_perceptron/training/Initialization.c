#include "../../stat/gaussian.h"
#include "../basic/basic.h"
#include "training.h"
#include <math.h>

double glorot(size_t fan_in, size_t fan_out) {
    double stddev = sqrt(2./ (fan_in + fan_out));
    return rand_gaussian(0, stddev);
}

double he(size_t fan_in) {
    double stddev = sqrt(2. / fan_in);
    return rand_gaussian(0, stddev);
}

void initialize_layer_glorot(Layer layer) {
    for (int i = 0; i < layer.incoming_weights.height; i++) {
        for (int j = 0; j < layer.incoming_weights.width; j++) {
            layer.incoming_weights.values[i * layer.incoming_weights.width + j] = glorot(layer.fan_in, layer.fan_out == NO_FAN ? 0 : layer.fan_out);
        }
    }
}

void initialize_layer_he(Layer layer) {
    for (int i = 0; i < layer.incoming_weights.height; i++) {
        for (int j = 0; j < layer.incoming_weights.width; j++) {
            layer.incoming_weights.values[i * layer.incoming_weights.width + j] = he(layer.fan_in);
        }
    }
}

void initialize_for_training(Network *network) {
    for (int i = 1; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        if (layer.func == RELU) {
            initialize_layer_he(layer);
        } else {
            initialize_layer_glorot(layer);
        }
    }
}



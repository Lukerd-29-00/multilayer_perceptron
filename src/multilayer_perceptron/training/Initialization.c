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

void initialize_layer_glorot(Layer *layer) {
    size_t fan_in = layer->prev->size;
    size_t fan_out = layer->size;

    for (int i = 0; i < layer->incoming_weights.height; i++) {
        for (int j = 0; j < layer->incoming_weights.width; j++) {
            layer->incoming_weights.values[i * layer->incoming_weights.width + j] = glorot(fan_in, fan_out);
        }
    }
}

void initialize_layer_he(Layer *layer) {
    size_t fan_in = layer->prev->size;

    for (int i = 0; i < layer->incoming_weights.height; i++) {
        for (int j = 0; j < layer->incoming_weights.width; j++) {
            layer->incoming_weights.values[i * layer->incoming_weights.width + j] = he(fan_in);
        }
    }
}

void initialize_for_training(Layer *first) {
    for (Layer *ptr = first->next; ptr != NULL; ptr = ptr->next) {
        if (ptr->func == RELU) {
            initialize_layer_he(ptr);
        } else {
            initialize_layer_glorot(ptr);
        }
    }
}



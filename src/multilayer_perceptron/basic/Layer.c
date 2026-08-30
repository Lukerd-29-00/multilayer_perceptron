#include "basic.h"
#include "..\..\linal\linal.h"
#include "..\..\activation\activation.h"
#include <stddef.h>
#include <stdlib.h>

Layer *initialize_first_layer(const size_t size) {
    Layer * const output = malloc(sizeof(Layer));
    if (output == NULL) {
        return NULL;
    }

    Matrix incoming_matrix;
    incoming_matrix.values = NULL;
    incoming_matrix.height = NO_INCOMING_WEIGHTS;
    incoming_matrix.width = NO_INCOMING_WEIGHTS;

    output->next = NULL;
    output->prev = NULL;
    output->size = size;
    output->incoming_weights = incoming_matrix;
    output->biases = NULL;
    output->func = NONE;
    return output;
}

Layer *initialize_layer(const size_t size, const Activation act_func, const size_t fan_in) {
    Layer * const output = malloc(sizeof(Layer));
    if (output == NULL) {
        return NULL;
    }
    
    
    double *biases = malloc(sizeof(double) * size);
    if (biases == NULL) {
        free(output);
        return NULL;
    }

    Matrix incoming_matrix;
    incoming_matrix.height = size;
    incoming_matrix.width = fan_in;
    incoming_matrix.values = malloc(sizeof(double) * incoming_matrix.width * incoming_matrix.height);
    if (incoming_matrix.values == NULL) {
        free(biases);
        free(output);
        return NULL;
    }
    

    output->next = NULL;
    output->prev = NULL;
    output->size = size;
    output->incoming_weights = incoming_matrix;
    output->biases = biases;
    output->func = act_func;
    return output;
}

void apply_transformation(const Layer * const layer, const double * restrict const input_vec, double * const restrict output_vec) {
    transform(&layer->incoming_weights, input_vec, output_vec);
    
    add(output_vec, layer->biases, layer->size);
}

void apply_activation(Activation func, double *transformed_vec, const size_t vector_size) {
    switch (func) {
        case SIGMOID:
            for (int i = 0; i < vector_size; i++) {
                transformed_vec[i] = sigmoid(transformed_vec[i]);
            }
        break;
        case RELU:
            for (int i = 0; i < vector_size; i++) {
                transformed_vec[i] = relu(transformed_vec[i]);
            }
        break;
        case SOFTMAX:
            softmax_destructive(transformed_vec, vector_size);
            break;
        case NONE:
            errno = EINVAL;
            break;
    }
}

void destroy_layer(Layer * layer) {
    if (layer->incoming_weights.values != NULL) {
        free(layer->incoming_weights.values);
    }
    if (layer->biases != NULL) {
        free(layer->biases);
    }
    
    free(layer);
}

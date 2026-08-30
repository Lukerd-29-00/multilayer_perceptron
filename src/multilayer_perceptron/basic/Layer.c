#include "basic.h"
#include "..\..\linal\linal.h"
#include "..\..\activation\activation.h"
#include <stddef.h>
#include <stdlib.h>

Layer *initialize_layer(size_t size, Activation act_func, size_t fan_in) {
    Layer *output = malloc(sizeof(Layer));
    if (output == NULL) {
        return NULL;
    }
    
    Vector biases;
    

    Matrix incoming_matrix;
    if (fan_in != NO_INCOMING_WEIGHTS) {

        biases.dimensions = size;
        biases.values = malloc(sizeof(double) * biases.dimensions);
        if (biases.values == NULL) {
            free(output);
            return NULL;
        }

        incoming_matrix.height = size;
        incoming_matrix.width = fan_in;
        incoming_matrix.values = malloc(sizeof(double) * incoming_matrix.width * incoming_matrix.height);
        if (incoming_matrix.values == NULL) {
            free(biases.values);
            free(output);
            return NULL;
        }
    } else {
        biases.values = NULL;
        biases.dimensions = NO_INCOMING_WEIGHTS;
        incoming_matrix.values = NULL;
        incoming_matrix.height = NO_INCOMING_WEIGHTS;
        incoming_matrix.width = NO_INCOMING_WEIGHTS;
    }

    output->next = NULL;
    output->prev = NULL;
    output->size = size;
    output->incoming_weights = incoming_matrix;
    output->biases = biases;
    output->func = act_func;
}

//Future optimization: assume that the input vector is the size of the biggest layer in the network, and leave irrelevant rows untouched to avoid allocating and freeing repeatedly.
Vector *apply_transformation(Layer *layer, Vector *input_vec) {
    Vector *output_vec = initialize_vec(layer->size);
    if (output_vec == NULL) {
        return NULL;
    }
    transform(&layer->incoming_weights, input_vec, output_vec);
    
    add(output_vec, &layer->biases);
    return output_vec;
}

Vector *apply_softmax_nondestructive(Vector *transformed_vec) {
    Vector *output_vec = initialize_vec(transformed_vec->dimensions);
    if (output_vec == NULL) {
        return NULL;
    }
    softmax(transformed_vec->values, transformed_vec->dimensions, output_vec->values);
    return output_vec;
}

void apply_activation_destructive(Activation func, Vector *transformed_vec) {
    switch (func) {
        case SIGMOID:
            for (int i = 0; i < transformed_vec->dimensions; i++) {
                transformed_vec->values[i] = sigmoid(transformed_vec->values[i]);
            }
        break;
        case RELU:
            for (int i = 0; i < transformed_vec->dimensions; i++) {
                transformed_vec->values[i] = relu(transformed_vec->values[i]);
            }
        break;
        case SOFTMAX:
            softmax_destructive(transformed_vec->values, transformed_vec->dimensions);
            break;
        case NONE:
            errno = EINVAL;
            break;
    }
}

Layer * destroy_layer(Layer * layer) {
    Layer *output = layer->next;
    free(layer->incoming_weights.values);
    free(layer->biases.values);
    free(layer);
    return output;
}

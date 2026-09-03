#include "basic.h"
#include "../../linal/linal.h"
#include "../../activation/activation.h"
#include <stddef.h>
#include <stdlib.h>

Layer initialize_first_layer(const size_t size, const size_t fan_out) {
    Matrix incoming_matrix;
    incoming_matrix.values = NULL;
    incoming_matrix.height = NO_INCOMING_WEIGHTS;
    incoming_matrix.width = NO_INCOMING_WEIGHTS;

    Layer output;
    output.size = size;
    output.incoming_weights = incoming_matrix;
    output.biases = NULL;
    output.func = NONE;
    output.fan_in = NO_FAN;
    output.fan_out = fan_out;

    return output;
}

Layer initialize_layer(const size_t size, const Activation act_func, const size_t fan_in, const size_t fan_out) {
    Layer output;
    double *biases = malloc(sizeof(double) * size);
    if (biases == NULL) {
        output.biases = NULL;
        return output;
    }

    Matrix incoming_matrix;
    incoming_matrix.height = size;
    incoming_matrix.width = fan_in;
    incoming_matrix.values = malloc(sizeof(double) * incoming_matrix.width * incoming_matrix.height);
    if (incoming_matrix.values == NULL) {
        free(biases);
        output.biases = NULL;
        return output;
    }

    
    output.size = size;
    output.incoming_weights = incoming_matrix;
    output.biases = biases;
    output.func = act_func;
    output.fan_in = fan_in;
    output.fan_out = fan_out;

    return output;
}

void apply_transformation(const Layer layer, const double * restrict const input_vec, double * const restrict output_vec) {
    transform(&layer.incoming_weights, input_vec, output_vec);
    
    add(output_vec, layer.biases, layer.size);
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

void destroy_layer(Layer layer) {
    if (layer.incoming_weights.values != NULL) {
        free(layer.incoming_weights.values); 
    }
    if (layer.biases != NULL) {
        free(layer.biases);
    }
}

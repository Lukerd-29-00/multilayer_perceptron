#include "training.h"
#include "../basic/basic.h"
#include "../../activation/activation.h"
#include <assert.h>

void apply_activation_nondestructive(const Activation func, const double * const transformed_vec, const size_t vector_size, double * const output_vec) {
    assert(func != NONE);
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
    }
}

void feed_forward_save_state(const Network * const network, const double * const input_values, Layer_Calcs_t * const states) {
    for (int i = 0; i < network->layers_array[0].size; i++) {
        //input values and output values are always the same pointer for layer 1 
        states[0].input_values[i] = input_values[i];
    }

    for (int i = 1; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        Layer_Calcs_t state = states[i];
        apply_transformation(layer, state_vec_1, state.input_vec);
        apply_activation_nondestructive(layer.func, state.input_vec, layer.size, state.output_vec);
    }
}
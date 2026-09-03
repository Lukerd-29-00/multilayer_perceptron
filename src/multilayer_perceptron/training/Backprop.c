#include "training.h"
#include "../../linal/linal.h"
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

thread_local Scratchpad_t scratchpad;

void initialize_backprop(Layer *network) {
    size_t largest_layer = 0;
    Layer *last;
    for (Layer *ptr = network; ptr != NULL; ptr = ptr->next) {
        if (ptr->size > largest_layer) {
            largest_layer = ptr->size;
        }
        if (ptr->next == NULL) {
            last = ptr;
        }
    }
    
    scratchpad.activation_derivative = malloc(sizeof(double) * largest_layer);
    if (scratchpad.activation_derivative == NULL) {
        return;
    }
    scratchpad.weight_transform = intialize_matrix(largest_layer, largest_layer);
    if (scratchpad.weight_transform == NULL) {
        free(scratchpad.activation_derivative);
        return;
    }
    scratchpad.cost_deriv_vec = malloc(sizeof(double) * largest_layer);
    if (scratchpad.cost_deriv_vec == NULL) {
        free(scratchpad.activation_derivative);
        free(scratchpad.weight_transform);
        return;
    }
}

void teardown_backprop() {
    free(scratchpad.activation_derivative);
    destroy_matrix(scratchpad.incoming_weights);
    free(scratchpad.cost_deriv_vec);
}



void calculate_backprop_for_run(const Network * const network, const Layer_Calcs_t * const calcs, const double *correct_answer, Backprop_Output_t *output) {
    size_t last_layer_idx = network->layers_count - 1;
    Layer last = network->layers_array[last_layer_idx];
    double *output_values = calcs[last_layer_idx].output_values;
    double *last_bias_derivs = output[last_layer_idx];
    double *last_inputs = calcs[last_layer_idx].input_values;
    derivative_of(SOFTMAX, last_inputs,  last.size, last_bias_derivs);

    for (int i = 0; i < last.size; i++) {
        double original_deriv = last_bias_derivs[i];
        last_bias_derivs[i] = 0;
        for (int j = 0; j < last.size; j++) {
            // d/dz_j SOFTMAX(original_input_value[i]) = - d/dz_j SOFTMAX(original_input_value[i])
            last_bias_derivs[i] += (j==i ? original_deriv : -original_deriv) * 2 * (output_values[j] - correct_answer[j])
        }
    }

    int i = last_layer_idx;
    do {

    } while (i > 0) //Skipping the first layer is intentional
}

void calculate_dc_dinput_hidden(const Layer * const hidden_layer, const double * const restrict input_values, const double * const restrict derivs, const Activation activation, double * const restrict output) {
    Layer current_layer = *hidden_layer;
    derivative_of(activation, input_values, current_layer.fan_in, scratchpad->activation_derivative);
    size_t height = scratchpad->weight_transform->height;
    size_t width = scratchpad->weight_transform->width;
    //This is required to be able to index this matrix properly, not just a hack around asserts!
    scratchpad->weight_transform->height = current_layer.incoming_weights.width;
    scratchpad->weight_transform->width = current_layer.incoming_weights.height;
    transpose(&current_layer.incoming_weights, scratchpad->weight_transform);
    scale_rows_destructive(scratchpad->weight_transform, scratchpad->activation_derivative);
    transform(scratchpad->weight_transform, derivs, output);
    scratchpad->weight_transform->height = height;
    scratchpad->weight_transform->width = width;
}

void calculate_weight_derivs(Layer *hidden_layer, const double * const restrict previous_outputs, double *derivs, Matrix *weight_outputs) {
    outer_product(derivs, layer->size, previous_outputs, layer->fan_in, weight_outputs);
}

void derivative_of(const Activation activation, const double * const restrict input_values, const size_t value_count, double * const restrict output_values) {
    assert(activation != NONE);
    
    switch (activation) {
        case SIGMOID:
            calculate_sigmoid_primes(input_values, output, value_count);
            break;
        case RELU:
            calculate_relu_primes(input_values, output, value_count);
            break;
        case SOFTMAX:
            calculate_softmax_prime(input_values, output, value_count);
            break;
    }
}

void calculate_sigmoid_primes(double *input_values, double *derivs, size_t count) {
    for (int i = 0; i < count; i++) {
        dervis[i] = sigmoid_prime(input_values[i]);
    }
}

void calculate_relu_primes(double *input_values, double *derivs, size_t count) {
    for (int i = 0; i < count; i++) {
        derivs[i] = relu_prime(input_values[i]);
    }
}

void calculate_softmax_prime(double *input_values, double *derivs, size_t count) {
    for (int i = 0; i < count; i++) {
        derivs[i] = softmax_prime(input_values, i, count);
    }
}
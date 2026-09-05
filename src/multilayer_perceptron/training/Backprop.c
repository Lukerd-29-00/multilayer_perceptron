#include "training.h"
#include "../../linal/linal.h"
#include "../basic/basic.h"
#include "../../activation/activation.h"
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

thread_local Scratchpad_t scratchpad;

void initialize_backprop(const Network * const network) {
    scratchpad.size = network->largest_layer_size;
    
    scratchpad.activation_derivative = malloc(sizeof(double) * scratchpad.size);
    if (scratchpad.activation_derivative == NULL) {
        return;
    }
    scratchpad.weight_transform = initialize_matrix(scratchpad.size, scratchpad.size);
    if (scratchpad.weight_transform == NULL) {
        free(scratchpad.activation_derivative);
        return;
    }
    
}

void teardown_backprop() {
    free(scratchpad.activation_derivative);
    destroy_matrix(scratchpad.weight_transform);
}

void mask_scratchpad_size(const size_t height, const size_t width) {
    scratchpad.weight_transform->height = height;
    scratchpad.weight_transform->width = width;
}

void revert_scratchpad_size() {
    scratchpad.weight_transform->height = scratchpad.size;
    scratchpad.weight_transform->width = scratchpad.size;
}


void calculate_backprop_for_run(const Network * const network, const Layer_Calcs_t * const calcs, const double *correct_answer, Backprop_Output_t *output) {
    size_t last_layer_idx = network->layers_count - 1;
    Layer last = network->layers_array[last_layer_idx];
    double *output_values = calcs[last_layer_idx].output_values;
    double *last_bias_derivs = output[last_layer_idx - 1].bias_derivs;

    calculate_softmax_prime(output_values, correct_answer, last_bias_derivs, last.size);

    for (int i = last_layer_idx; i > 1; i--) { //Skipping the first hidden layer is intentional
        
        //output[i-1] corresponds to layer i, because the input layer has no weights
        calculate_weight_derivs(&(network->layers_array[i]), calcs[i-1].output_values, output[i-1].bias_derivs, &(output[i-1].weight_derivs));

        calculate_dc_dinput_hidden(&(network->layers_array[i]), calcs[i-1].input_values, output[i-1].bias_derivs, network->layers_array[i-1].func, output[i-2].bias_derivs);
    }

    calculate_weight_derivs(&(network->layers_array[1]), calcs[0].output_values, output[1].bias_derivs, &(output[1].weight_derivs));
}

void calculate_dc_dinput_hidden(const Layer * const hidden_layer, const double * const restrict input_values, const double * const restrict derivs, const Activation activation, double * const restrict output) {
    Layer current_layer = *hidden_layer;
    derivative_of(activation, input_values, current_layer.fan_in, scratchpad.activation_derivative);
    mask_scratchpad_size(current_layer.incoming_weights.height, current_layer.incoming_weights.width);
    
    transpose(&current_layer.incoming_weights, scratchpad.weight_transform);
    scale_rows_destructive(scratchpad.weight_transform, scratchpad.activation_derivative);
    transform(scratchpad.weight_transform, derivs, output);
    scratchpad.weight_transform->height = height;
    scratchpad.weight_transform->width = width;

    revert_scratchpad_size();
}

void calculate_weight_derivs(Layer *layer, const double * const restrict previous_outputs, double *derivs, Matrix *weight_outputs) {
    outer_product(derivs, layer->size, previous_outputs, layer->fan_in, weight_outputs);
}

void derivative_of(const Activation activation, const double * const restrict input_values, const size_t value_count, double * const restrict output_values) {
    assert(activation != NONE);
    assert(activation != SOFTMAX);
    
    switch (activation) {
        case SIGMOID:
            calculate_sigmoid_primes(input_values, output_values, value_count);
            break;
        case RELU:
            calculate_relu_primes(input_values, output_values, value_count);
            break;
    }
}

void calculate_sigmoid_primes(const double * const restrict input_values, double * const restrict derivs, const size_t count) {
    for (int i = 0; i < count; i++) {
        derivs[i] = sigmoid_prime(input_values[i]);
    }
}

void calculate_relu_primes(const double * const restrict input_values, double * const restrict derivs, const size_t count) {
    for (int i = 0; i < count; i++) {
        derivs[i] = relu_prime(input_values[i]);
    }
}

void calculate_softmax_prime(const double * const restrict input_values, const double * const restrict real_values, double * const restrict derivs, const size_t count) {
    mask_scratchpad_size(count, count);
    softmax_prime(input_values, scratchpad.weight_transform);

    for (int i = 0; i < count; i++) {
        scratchpad.activation_derivative[i] = 2 * (input_values[i] - real_values[i]);
    }

    transform(scratchpad.weight_transform, scratchpad.activation_derivative, derivs);

    revert_scratchpad_size();
}

void update_network(Network * const network, const Backprop_Output_t * const back_propogation_outputs) {
    for (int i = 1; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        Backprop_Output_t output = back_propogation_outputs[i-1];
        add(layer.biases, output.bias_derivs, layer.size);
        add_matrices(&layer.incoming_weights, &output.weight_derivs);
    }
}
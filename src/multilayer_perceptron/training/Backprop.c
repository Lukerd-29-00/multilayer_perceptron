#include "training.h"
#include "../../linal/linal.h"
#include <errno.h>
#include <stdlib.h>

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



void calculate_backprop_for_run(const Layer * const network, const Layer_Calcs_t * const calcs, const double *correct_answer, Backprop_Output_t *output) {
    Layer *last;
    size_t i = 0;
    for (last = network; last->next != NULL; last = last->next) {
        i++;
    }
    for (int i = 0; i < last->size; i++) {
        throwaway[i] = 2 * (calcs[i].output_values - correct_answer[i]);
    }
    

}

void calculate_for_layer(const Layer * const layer, const Layer_Calcs_t * const calcs, size_t fan_out, double *previous_derivs) {
    switch (layer->func) {
        case SIGMOID:
            calculate_sigmoid_primes(calcs->input_values, scratchpad->activation_derivative);
            break;
        case RELU:
            calculate_relu_primes(calcs->input_values, scratchpad->activation_derivative);
            break;
        case SOFTMAX:
        case NONE:
            errno = EINVAL;
            return;
    }
    transpose(&layer->incoming_weights, scratchpad->weight_transform);
    scale_rows_destructive(scratchpad->weight_transform, scratchpad->activation_derivative);
    transform(scratchpad->weight_transform, previous_derivs, output->bias_derivs);
    scale_rows(&layer->incoming_weights, output->bias_derivs, output->weight_derivs);
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

void update_network(Layer *network, Matrix **backprop_derivs, double **bias_derivs);



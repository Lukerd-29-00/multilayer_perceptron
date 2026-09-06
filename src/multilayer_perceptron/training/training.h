#include "../../linal/linal.h"
#include "../basic/basic.h"
#include <stddef.h>

#ifndef TRAINING_TYPES
#define TRAINING_TYPES

typedef struct memo {
    double *input_values;
    double *output_values;
} Layer_Calcs_t;

typedef struct back {
    Matrix *weight_derivs;
    double *bias_derivs;
} Backprop_Output_t;

typedef struct scratch {
    double *activation_derivative;
    Matrix *weight_transform;
    size_t size;
} Scratchpad_t;

#endif

double glorot(size_t fan_in, size_t fan_out);
double he(size_t fan_in);
void initialize_layer_glorot(Layer layer);
void initialize_layer_he(Layer layer);
void initialize_for_training(Network *network);


void initialize_backprop(const size_t largest_size);
void teardown_backprop();
void mask_scratchpad_size(const size_t height, const size_t width);
void revert_scratchpad_size();
void calculate_backprop_for_run(const Network * const network, const Layer_Calcs_t * const calcs, const double *correct_answer, Backprop_Output_t *output);
void calculate_dc_dinput_hidden(const Layer * const hidden_layer, const double * const restrict input_values, const double * const restrict derivs, const Activation activation, double * const restrict output);
void calculate_weight_derivs(Layer *layer, const double * const restrict previous_outputs, double *derivs, Matrix *weight_outputs);
void derivative_of(const Activation activation, const double * const restrict input_values, const size_t value_count, double * const restrict output_values);
void calculate_sigmoid_primes(const double * const restrict input_values, double * const restrict derivs, const size_t count);
void calculate_relu_primes(const double * const restrict input_values, double * const restrict derivs, const size_t count);
void calculate_softmax_prime(const double * const restrict input_values, const double * const restrict real_values, double * const restrict derivs, const size_t count);
void update_network(Network * const network, const Backprop_Output_t * const back_propogation_outputs);
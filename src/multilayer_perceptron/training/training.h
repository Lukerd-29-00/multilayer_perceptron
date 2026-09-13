#include "../../linal/linal.h"
#include "../basic/basic.h"
#include <stddef.h>

#ifndef TRAINING_TYPES
#define TRAINING_TYPES

#define FAILED_CONNECTION 1005

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

typedef struct training_run {
    double *input_values;
    double *correct_answer;
} Training_Run_t;

#endif

//Backprop.c Public interface
void initialize_backprop(const size_t largest_size);
Backprop_Output_t *initialize_backprop_output(Layer *layer);
Backprop_Output_t *initialize_backprop_outputs(Network *network);
void update_network(Network * const network, const Backprop_Output_t * const back_propogation_outputs);
void teardown_backprop();
void teardown_backprop_output(Backprop_Output_t *backprop_output);
void teardown_backprop_outputs(Backprop_Output_t *backprop_outputs, size_t size);

//Backprop.c functions
void mask_scratchpad_size(const size_t height, const size_t width);
void revert_scratchpad_size();
void calculate_backprop_for_run(const Network * const network, const Layer_Calcs_t * const calcs, const double *correct_answer, Backprop_Output_t *output);
void calculate_dc_dinput_hidden(const Layer * const hidden_layer, const double * const restrict input_values, const double * const restrict derivs, const Activation activation, double * const restrict output);
void calculate_weight_derivs(Layer *layer, const double * const restrict previous_outputs, double *derivs, Matrix *weight_outputs);
void derivative_of(const Activation activation, const double * const restrict input_values, const size_t value_count, double * const restrict output_values);
void calculate_sigmoid_primes(const double * const restrict input_values, double * const restrict derivs, const size_t count);
void calculate_relu_primes(const double * const restrict input_values, double * const restrict derivs, const size_t count);
void calculate_softmax_prime(const double * const restrict input_values, const double * const restrict real_values, double * const restrict derivs, const size_t count);

//Initialization.c Public interface
void initialize_for_training(Network *network);

//Initialization.c functions
double glorot(size_t fan_in, size_t fan_out);
double he(size_t fan_in);
void initialize_layer_glorot(Layer layer);
void initialize_layer_he(Layer layer);

//Training.c Public interface
void train(Network *network, const char *data_file, const char *learnset, const int sample_size, const int runs);

//Training.c functions
void set_backprop_output_zero(Network *network, Backprop_Output_t *output);
void sum_to_avg(const Network *network, Backprop_Output_t *output, size_t sample_size);
Layer_Calcs_t *initialize_calcs(Network *network);
Training_Run_t *initialize_training_runs(Network *network, size_t population_size, size_t sample_count);
void teardown_calcs(Layer_Calcs_t *calcs, size_t size);
int sample_boundary(const int pop_size, const int sample_size, const int idx);
void apply_activation_training(Activation func, const double * restrict transformed_vec, const size_t vector_size, double * restrict output);
void feed_forward_training(const Network *network, double *input_values, Layer_Calcs_t *calcs);
void train_on_sample(const Network *network, const Training_Run_t *sample, const size_t sample_size, Layer_Calcs_t *calcs, Backprop_Output_t *backprop_output, Backprop_Output_t *avg_backprop);
void teardown_training_runs(Training_Run_t *training_runs, size_t population_size, size_t sample_count);
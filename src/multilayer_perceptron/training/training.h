#include "../../linal/linal.h"
#include "../basic/basic.h"
#include "../../linal/linal.h"
#include <stddef.h>

#ifndef TRAINING_TYPES
#define TRAINING_TYPES

typedef struct memo {
    double *input_values;
    double *output_values;
} Layer_Calcs_t;

typedef struct back {
    Matrix weight_derivs;
    double *bias_derivs;
} Backprop_Output_t;

typedef struct scratch {
    double *activation_derivative;
    double *cost_deriv_vec;
    Matrix *weight_transform;
} Scratchpad_t;

#endif

double glorot(size_t fan_in, size_t fan_out);
double he(size_t fan_in);
void initialize_layer_glorot(Layer layer);
void initialize_layer_he(Layer layer);
void initialize_for_training(Network *network);
void update_network(Layer *network, Matrix **backprop_derivs, double **bias_derivs);

/**
 * @brief calculates the back propogation of the network for a training run
 *
 * Calculates the partial derivative of sigma((Ri - zi)**2) for every real value R and output value z for the training run
 * All memory used by this function is pre-allocated before training begins for perfomrance reasons.
 *
 * @param network The pointer to the start of the neural network
 * @param calcs The calculated values from a training run (input, output for each layer)
 * @param correct_answer the correct answer from the training data
 * @param output An array of output structures that each hold a bias vector and a weight matrix. These are the result of the function.
 * @param throwaway An array used to calculate dC/da for each input to the final layer (C is the cost). Must be the size of the output layer or larger.
 * @param scratchpad structure of a pre-allocated matrix and vector used for some calculations. Matrix must be the max layer height x the max layer width. activation_derivative should be the size of the largest layer in the network.
 * 
 */
void calculate_backprop_for_run(const Layer * const network, const Layer_Calcs_t * const calcs, const double *correct_answer, Backprop_Output_t *output);


/**
 * @brief calculates the backprop values for this layer
 * 
 * Takes a layer and calculates the derivative for the incoming weights, and the derivative for the biases of the preceding layer.
 * Must be a hidden layer; i.e. not softmax and not the input layer.
 *
 * @param layer The layer whose incoming weights
 * @param calcs The stashed values from forward propogation
 * @param output_values The output values from the layer during forward propogation
 * @param deriv_values The calculated bias derivatives from the following layer
 * @param fan_out The size of the output vector
 * @param output_matrix The matrix to output the derivatives of the incoming weights
 * @param output_biases The vector to write the calculated biases to
*/
void calculate_for_layer(const Layer * const layer, const Layer_Calcs_t * const calcs, size_t fan_out, double *previous_derivs);

/**
 * @brief Calulates the derivative of cost / inputs to the layer before hidden_layer.
 * 
 * Calulates the partial derivative of the cost function relative to the input to each neuron in the previous layer
 * Assumes initialize_backprop has been invoked in this thread. 
 * The derivative of the biases is equivalent to this value.
 * 
 * @param hidden_layer The layer after the one we are calculating the derivative of
 * @param input_values The input values to the previous layer
 * @param derivs The derivatives with respect to the inputs of this layer
 * @param activation The activation function of the previous layer
 * @param output The array to write the answer to
 */
void calculate_dc_dinput_hidden(const Layer * const hidden_layer, const double * const restrict input_values, const double * const restrict derivs, const Activation activation, double * const restrict output);

void calculate_sigmoid_primes(double *input_values, double *derivs, size_t count);
void calculate_relu_primes(double *input_values, double *derivs, size_t count);
void calculate_softmax_prime(double *input_values, double *derivs, size_t count);

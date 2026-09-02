#include "..\..\linal\linal.h"
#include <stddef.h>
#include <stdbool.h>

#ifndef PERCEPTRON_TYPES
#define PERCEPTRON_TYPES

#define NO_INCOMING_WEIGHTS -1
typedef enum {SIGMOID, RELU, SOFTMAX, NONE} Activation;

//Optimization: make a network into an array instead of a DLL
typedef struct lay {
    size_t size;
    Matrix incoming_weights;
    double * restrict biases;
    Activation func;
    size_t fan_in;
    size_t fan_out;
} Layer;

typedef struct lay_inf {
    size_t size;
     Activation activation;
} Layer_Info_t;

typedef struct net {
    size_t layers_count;
    size_t largest_layer_size;
    Layer *layers_array;
} Network;

#define NO_FAN -1

#endif

Layer initialize_first_layer(const size_t size, const size_t fan_out);
Layer initialize_layer(const size_t size, const Activation act_func, const size_t fan_in, const size_t fan_out);
void apply_transformation(const Layer layer, const double * restrict const input_vec, double * const restrict output_vec);
void apply_activation(Activation func, double *transformed_vec, const size_t vector_size);
void destroy_layer(Layer layer);
Network *build_network(size_t input_size, Layer_Info_t *layer_info, size_t active_layer_count);
double *feed_forward(const Network * const network, double * const input_vec, double * restrict state_vec_1, double * restrict state_vec_2);
void destroy_network(Network *network);

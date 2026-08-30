#include "..\..\linal\linal.h"
#include <stddef.h>
#include <stdbool.h>

#ifndef PERCEPTRON_TYPES
#define PERCEPTRON_TYPES

#define NO_INCOMING_WEIGHTS -1
typedef enum {SIGMOID, RELU, SOFTMAX, NONE} Activation;

typedef struct lay {
    struct lay *next;
    struct lay *prev;
    size_t size;
    Matrix incoming_weights;
    double * restrict biases;
    Activation func;
} Layer;

typedef struct lay_inf {
    size_t size;
     Activation activation;
} Layer_Info_t;

#endif

Layer *initialize_layer(const size_t size, const Activation act_func, const size_t fan_in);
double *apply_transformation(const Layer * const layer, const double * restrict const input_vec);
void apply_activation(Activation func, double *transformed_vec, const size_t vector_size);
Layer *destroy_layer(Layer * layer);
Layer *build_network(size_t input_size, Layer_Info_t *layer_info, size_t layer_count);
double *feed_forward(const Layer * const network, double * const input_vec);
void destroy_network(Layer *first);

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
    Vector biases;
    Activation func;
} Layer;

typedef struct lay_inf {
    size_t size;
    Activation activation;
} Layer_Info_t;

#endif

Layer *initialize_layer(size_t size, Activation act_func, size_t fan_in);
Vector *apply_transformation(Layer *layer, Vector *input_vec);
Vector *apply_softmax_nondestructive(Vector *transformed_vec);
void apply_activation_destructive(Activation func, Vector *transformed_vec);
Layer *destroy_layer(Layer * layer);
Layer *build_network(size_t input_size, Layer_Info_t *layer_info, size_t layer_count);
Vector *feed_forward(const Layer * const network, Vector *input_vec);
void destroy_network(Layer *first);
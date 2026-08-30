#include "..\..\Testing.h"
#include "..\..\..\..\src\multilayer_perceptron\basic\basic.h"
#include "..\..\..\..\src\linal\linal.h"
#include <stddef.h>


char *test_initialize_layer(void) {
    Layer *layer = initialize_layer(3, SIGMOID, 2);
    if (layer == NULL) {
        return NULL;
    }
    char *error_message = NULL;
    ASSERT_NULL(layer->prev, "initialize layer");
    ASSERT_NULL(layer->next, "initialize layer");
    ASSERT_INT_EQUALS(3, layer->size, "initialize layer");
    ASSERT_INT_EQUALS(3, layer->incoming_weights.height, "initialize layer");
    ASSERT_INT_EQUALS(2, layer->incoming_weights.width, "intialize layer");
    ASSERT_INT_EQUALS(3, layer->biases.dimensions, "initialize layer");
    ASSERT_ACTIVATION_EQUALS(SIGMOID, layer->func, "initialize layer");


    destroy_layer(layer);
}

char *test_apply_transformation(void) {
    Layer *layer = initialize_layer(2, SIGMOID, 2);
    if (layer == NULL) {
        return NULL;
    }
    char *error_message = NULL;
    for (int i = 0; i < 2; i++) {
        layer->biases.values[i] = i;
    }
    for (int i = 0; i < 4; i++) {
        layer->incoming_weights.values[i] = 2;
    }
    Vector *input_vec = initialize_vec(2);
    if (input_vec == NULL) {
        destroy_layer(layer);
        return NULL;
    }
    for (int i = 0; i < input_vec->dimensions; i++) {
        input_vec->values[i] = i + 1;
    }
    Vector *output_vec = apply_transformation(layer, input_vec);
    ASSERT_DOUBLE(6, output_vec->values[0], 0.01, "apply transformation");
    ASSERT_DOUBLE(7, output_vec->values[1], 0.01, "apply transformation");

    destroy_layer(layer);
    destroy_vector(input_vec);
    destroy_vector(output_vec);
    return NULL;
}

char *test_apply_softmax_nondestructive(void) {
    Vector *input_vec = initialize_vec(3);
    if (input_vec == NULL) {
        return NULL;
    }
    for (int i = 0; i < 3; i++) {
        input_vec->values[i] = i + 1;
    }
    char *error_message = NULL;

    Vector *output_vec = apply_softmax_nondestructive(input_vec);
    ASSERT_DOUBLE(0.09003057317038045, output_vec->values[0], 0.01, "apply softmax non-destructive");
    ASSERT_DOUBLE(0.24472847105479764, output_vec->values[1], 0.01, "apply softmax non-destructive");
    ASSERT_DOUBLE(0.6652409557748219, output_vec->values[2], 0.01, "apply softmax non-destructive");
    
    destroy_vector(output_vec);
    destroy_vector(input_vec);
    return NULL;
}

char *test_apply_activation_destructive_sigmoid(void) {
    Vector *input_vec = initialize_vec(2);
    if (input_vec == NULL) {
        return NULL;
    }
    char *error_message = NULL;
    input_vec->values[0] = 6;
    input_vec->values[1] = 7;
    
    apply_activation_destructive(SIGMOID, input_vec);

    ASSERT_DOUBLE(0.9975273768433653, input_vec->values[0], 0.01, "destructive sigmoid");
    ASSERT_DOUBLE(0.9990889488055994, input_vec->values[1], 0.01, "destructive sigmoid");

    destroy_vector(input_vec);
    return NULL;
}

char *test_apply_activation_destructive_relu(void) {
    Vector *input_vec = initialize_vec(2);
    if (input_vec == NULL) {
        return NULL;
    }
    char *error_message = NULL;
    input_vec->values[0] = 6;
    input_vec->values[1] = -1;
    
    apply_activation_destructive(RELU, input_vec);

    ASSERT_DOUBLE(6, input_vec->values[0], 0.01, "destructive relu");
    ASSERT_DOUBLE(0, input_vec->values[1], 0.01, "destructive relu");

    destroy_vector(input_vec);
    return NULL;
}

char *test_apply_activation_destructive_softmax(void) {
    Vector *input_vec = initialize_vec(3);
    if (input_vec == NULL) {
        return NULL;
    }
    char *error_message = NULL;
    for (int i = 0; i < 3; i++) {
        input_vec->values[i] = i + 1;
    }

    apply_activation_destructive(SOFTMAX, input_vec);

    ASSERT_DOUBLE(0.09003057317038045, input_vec->values[0], 0.01, "apply softmax destructive");
    ASSERT_DOUBLE(0.24472847105479764, input_vec->values[1], 0.01, "apply softmax destructive");
    ASSERT_DOUBLE(0.6652409557748219, input_vec->values[2], 0.01, "apply softmax destructive");

    return NULL;
}
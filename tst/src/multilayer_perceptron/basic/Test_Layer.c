#include "..\..\Testing.h"
#include "..\..\..\..\src\multilayer_perceptron\basic\basic.h"
#include "..\..\..\..\src\linal\linal.h"
#include <stddef.h>
#include <stdlib.h>


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
    ASSERT_NOT_NULL(layer->biases, "initialize layer");
    ASSERT_ACTIVATION_EQUALS(SIGMOID, layer->func, "initialize layer");

    destroy_layer(layer);
    return NULL;
}

char *test_apply_transformation(void) {
    Layer *layer = initialize_layer(2, SIGMOID, 2);
    if (layer == NULL) {
        return NULL;
    }
    char *error_message = NULL;
    for (int i = 0; i < 2; i++) {
        layer->biases[i] = i;
    }
    for (int i = 0; i < 4; i++) {
        layer->incoming_weights.values[i] = 2;
    }
    double input_vec[2];
    for (int i = 0; i < 2; i++) {
        input_vec[i] = i + 1;
    }
    double *output_vec = apply_transformation(layer, input_vec);
    ASSERT_DOUBLE(6, output_vec[0], 0.01, "apply transformation");
    ASSERT_DOUBLE(7, output_vec[1], 0.01, "apply transformation");

    destroy_layer(layer);
    free(output_vec);
    return NULL;
}

char *test_apply_activation_sigmoid(void) {
    double input_vec[2];
    char *error_message = NULL;
    input_vec[0] = 6;
    input_vec[1] = 7;
    
    apply_activation(SIGMOID, input_vec, 2);

    ASSERT_DOUBLE(0.9975273768433653, input_vec[0], 0.01, "destructive sigmoid");
    ASSERT_DOUBLE(0.9990889488055994, input_vec[1], 0.01, "destructive sigmoid");

    return NULL;
}

char *test_apply_activation_relu(void) {
    double input_vec[2];
    char *error_message = NULL;
    input_vec[0] = 6;
    input_vec[1] = -1;
    
    apply_activation(RELU, input_vec, 2);

    ASSERT_DOUBLE(6, input_vec[0], 0.01, "destructive relu");
    ASSERT_DOUBLE(0, input_vec[1], 0.01, "destructive relu");

    return NULL;
}

char *test_apply_activation_softmax(void) {
    double input_vec[3];
    char *error_message = NULL;
    for (int i = 0; i < 3; i++) {
        input_vec[i] = i + 1;
    }

    apply_activation(SOFTMAX, input_vec, 3);

    ASSERT_DOUBLE(0.09003057317038045, input_vec[0], 0.01, "apply softmax destructive");
    ASSERT_DOUBLE(0.24472847105479764, input_vec[1], 0.01, "apply softmax destructive");
    ASSERT_DOUBLE(0.6652409557748219, input_vec[2], 0.01, "apply softmax destructive");

    return NULL;
}

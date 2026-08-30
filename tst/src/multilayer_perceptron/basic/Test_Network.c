#include "test_basic.h"
#include "..\..\..\..\src\multilayer_perceptron\basic\basic.h"
#include "..\..\..\..\src/linal/linal.h"
#include "..\..\Testing.h"


char *test_build_network(void) {
    Layer_Info_t layers[2];
    
    Layer_Info_t layer_two;
    layer_two.size = 3;
    layer_two.activation = RELU;

    Layer_Info_t layer_three;
    layer_three.size = 2;
    layer_three.activation = SOFTMAX;

    
    layers[0] = layer_two;
    layers[1] = layer_three;

    Layer *start = build_network(1, layers, 2);

    char *error_message = NULL;

    ASSERT_NOT_NULL(start, "build network");
    ASSERT_NULL(start->prev, "build network");
    ASSERT_INT_EQUALS(1, start->size, "build network");
    ASSERT_INT_EQUALS(NO_INCOMING_WEIGHTS, start->incoming_weights.height, "build network");
    ASSERT_INT_EQUALS(NO_INCOMING_WEIGHTS, start->incoming_weights.width, "build network");
    ASSERT_NULL(start->incoming_weights.values, "build network");
    ASSERT_INT_EQUALS(NO_INCOMING_WEIGHTS, start->biases.dimensions, "build network");
    ASSERT_NULL(start->biases.values, "build network");
    ASSERT_ACTIVATION_EQUALS(NONE, start->func, "build network");

    Layer *hidden = start->next;
    ASSERT_NOT_NULL(hidden, "build network");
    ASSERT_POINTER_EQUALS(start, hidden->prev, "build network");
    ASSERT_INT_EQUALS(2, hidden->size, "build network");
    ASSERT_INT_EQUALS(3, hidden->incoming_weights.height, "build network");
    ASSERT_INT_EQUALS(1, hidden->incoming_weights.width, "build network");
    ASSERT_NOT_NULL(hidden->incoming_weights.values, "build network");
    ASSERT_INT_EQUALS(3, hidden->biases.dimensions, "build network");
    ASSERT_NOT_NULL(hidden->biases.values, "build network");
    ASSERT_ACTIVATION_EQUALS(RELU, hidden->func, "build network");

    Layer *last = hidden->next;
    ASSERT_NOT_NULL(last, "build network");
    ASSERT_POINTER_EQUALS(hidden, last->prev, "build network");
    ASSERT_INT_EQUALS(2, last->size, "build network");
    ASSERT_INT_EQUALS(2, last->incoming_weights.height, "build network");
    ASSERT_INT_EQUALS(3, last->incoming_weights.width, "build network");
    ASSERT_NOT_NULL(last->incoming_weights.values, "build network");
    ASSERT_INT_EQUALS(2, last->biases.dimensions, "build network");
    ASSERT_NOT_NULL(last->biases.values, "build network");
    ASSERT_ACTIVATION_EQUALS(SOFTMAX, last->func, "build network");
    
    destroy_network(start);

    return NULL;
}

char *test_feed_forward(void) {

    Layer_Info_t layers[2];

    Layer_Info_t layer_two;
    layer_two.size = 3;
    layer_two.activation = RELU;

    Layer_Info_t layer_three;
    layer_three.size = 2;
    layer_three.activation = SOFTMAX;

    layers[0] = layer_two;
    layers[1] = layer_three;

    Layer *start = build_network(1, layers, 2);

    Vector *input_vec = initialize_vec(1);
    input_vec->values[0] = 1;

    Layer *hidden = start->next;
    double * hidden_weights = hidden->incoming_weights.values;
    hidden_weights[0] = 3;
    hidden_weights[1] = -10;
    hidden_weights[2] = 0.5;

    double *hidden_biases = hidden->biases.values;
    hidden_biases[0] = 0.1;
    hidden_biases[1] = 1;
    hidden_biases[2] = 3.14;

    Layer *last = hidden->next;

    double *last_weights = last->incoming_weights.values;
    last_weights[0] = 1.5;
    last_weights[1] = 1;
    last_weights[2] = 4.5;
    last_weights[3] = -2.33;
    last_weights[4] = -3;
    last_weights[5] = 4.5;

    last->biases.values[0] = -2.5;
    last->biases.values[1] = 3;

    Vector *output_vec = feed_forward(start, input_vec);

    char *error_message = NULL;

    ASSERT_INT_EQUALS(2, output_vec->dimensions, "feed forward");
    ASSERT_DOUBLE(2.681003867781731e-14, output_vec->values[0], 0.01, "feed forward");
    ASSERT_DOUBLE(0.9999999999999731, output_vec->values[1], 0.01, "feed forward");

    destroy_vector(output_vec);
    destroy_vector(input_vec);
    destroy_network(start);

    return NULL;
}
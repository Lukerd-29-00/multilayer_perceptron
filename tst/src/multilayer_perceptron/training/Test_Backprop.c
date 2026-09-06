#include "test_training.h"
#include "../../Testing.h"
#include "../../../../src/multilayer_perceptron/training/training.h"
#include "../../../../src/linal/linal.h"
#include "../../../../src/multilayer_perceptron/basic/basic.h"
#include <string.h>



char *test_calculate_backprop_for_run(void) {
    initialize_backprop(3);

    char *error_message = NULL;

    Layer_Info_t layers[3] = {
        {.size = 3, .activation = RELU},
        {.size = 3, .activation = SIGMOID},
        {.size = 2, .activation = SOFTMAX}
    };

    Network *network = build_network(2, layers, 3);
    double weight_values_1[6] = {-1.639, 0.381, 1.471, -0.479, -0.124, 0.001};
    double biases_1[3] = {0.1, -0.11, 0.1};

    double weight_values_2[9] = {-0.032, -0.158, 0.4, -0.705, -0.466, -0.382, 0.173, -0.511, 0.301};
    double biases_2[3] = {0.1, -0.11, 0.1};

    double weight_values_3[6] = {0.317, -0.866, 0.019, -0.559, -0.529, 0.065};
    double biases_3[2] = {0.1, -0.11};

    memcpy(network->layers_array[1].incoming_weights.values, weight_values_1, 6 * sizeof(double));
    memcpy(network->layers_array[1].biases, biases_1, 3 * sizeof(double));

    memcpy(network->layers_array[2].incoming_weights.values, weight_values_2, 9 * sizeof(double));
    memcpy(network->layers_array[2].biases, biases_2, 3 * sizeof(double));

    memcpy(network->layers_array[3].incoming_weights.values, weight_values_3, 6 * sizeof(double));
    memcpy(network->layers_array[3].biases, biases_3, 2 * sizeof(double));


    double input_values_1[2] = {1., 2.};

    double input_values_2[3] =  {-0.777,  0.403, -0.022};
    double output_values_2[3] = {0., 0.403, 0.};

    double input_values_3[3] = {0.036326, -0.297798, -0.105933};
    double output_values_3[3] = {0.5090805 , 0.42609587, 0.47354149};

    double input_values_4[2] = {-0.09862321, -0.58920052};
    double output_values_4[2] = {0.62024242, 0.37975758};

    Layer_Calcs_t run_calcs[4] = {
        {.input_values= input_values_1, .output_values= input_values_1},
        {.input_values= input_values_2, .output_values = output_values_2},
        {.input_values = input_values_3, .output_values = output_values_3},
        {.input_values = input_values_4, .output_values = output_values_4}
    };

    double weight_derivs_values_1[6];
    double bias_derivs_1[3];

    Matrix weights_1 = {.height = 3., .width = 2., .values = weight_derivs_values_1};

    double weight_derivs_values_2[9];
    double bias_derivs_2[3];
    Matrix weights_2 = {.height = 3, .width = 3, .values = weight_derivs_values_2};

    double weight_derivs_values_3[6];
    double bias_derivs_3[2];
    Matrix weights_3 = {.height = 2, .width = 3, .values = weight_derivs_values_3};

    Backprop_Output_t outputs[3] = {
        {.weight_derivs = &weights_1, .bias_derivs = bias_derivs_1},
        {.weight_derivs = &weights_2, .bias_derivs = bias_derivs_2},
        {.weight_derivs = &weights_3, .bias_derivs = bias_derivs_3}
    };

    double correct_answer[2] = {2, 1};

    calculate_backprop_for_run(network, run_calcs, correct_answer, outputs);

    ASSERT_DOUBLE(0, outputs[0].weight_derivs->values[0], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[0].weight_derivs->values[1], 0.01, "test run backprop");
    ASSERT_DOUBLE(-0.00346067, outputs[0].weight_derivs->values[2], 0.01, "test run backprop");
    ASSERT_DOUBLE(-0.00692134, outputs[0].weight_derivs->values[3], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[0].weight_derivs->values[4], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[0].weight_derivs->values[5], 0.01, "test run backprop");

    ASSERT_DOUBLE(0, outputs[0].bias_derivs[0], 0.01, "test run backprop");
    ASSERT_DOUBLE(-0.00346067, outputs[0].bias_derivs[1], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[0].bias_derivs[2], 0.01, "test run backprop");

    ASSERT_DOUBLE(0, outputs[1].weight_derivs->values[0], 0.01, "test run backprop");
    ASSERT_DOUBLE(-0.0315675, outputs[1].weight_derivs->values[1], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[1].weight_derivs->values[2], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[1].weight_derivs->values[3], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.01188272, outputs[1].weight_derivs->values[4], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[1].weight_derivs->values[5], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[1].weight_derivs->values[6], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.00165356, outputs[1].weight_derivs->values[7], 0.01, "test run backprop");
    ASSERT_DOUBLE(0, outputs[1].weight_derivs->values[8], 0.01, "test run backprop");

    ASSERT_DOUBLE(-0.07833127642559062, outputs[1].bias_derivs[0], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.029485665032752444, outputs[1].bias_derivs[1], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.004103121433836422, outputs[1].bias_derivs[2], 0.01, "test run backprop");

    ASSERT_DOUBLE(-0.18214649, outputs[2].weight_derivs->values[0], 0.01, "test run backprop");
    ASSERT_DOUBLE(-0.152455, outputs[2].weight_derivs->values[1], 0.01, "test run backprop");
    ASSERT_DOUBLE(-0.16943081, outputs[2].weight_derivs->values[2], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.18214649, outputs[2].weight_derivs->values[3], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.152455, outputs[2].weight_derivs->values[4], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.16943081, outputs[2].weight_derivs->values[5], 0.01, "test run backprop");

    ASSERT_DOUBLE(-0.35779507, outputs[2].bias_derivs[0], 0.01, "test run backprop");
    ASSERT_DOUBLE(0.35779507, outputs[2].bias_derivs[1], 0.01, "test run backprop");

    destroy_network(network);
    teardown_backprop();
    return NULL;
}

char *test_calculate_dc_dinput_hidden(void) {
    initialize_backprop(2);
    char *error_message = NULL;
    double layer_weight_values[4] = {1., 2., 3., 4.};
    double bias[2] = {1., 2.};
    Layer layer = {.size = 2, .incoming_weights = {.width = 2, .height = 2, .values = layer_weight_values}, .biases = bias, .func = SIGMOID, .fan_in = 2, .fan_out = 3};

    double derivs[2] = {0.12, -0.11};
    double input_values[2] = {0.15, -0.26};

    double new_derivs[2];

    calculate_dc_dinput_hidden(&layer, input_values, derivs, SIGMOID, new_derivs);

    ASSERT_DOUBLE(-0.05220579140228403, new_derivs[0], 0.01, "test calculate dc/dinput");
    ASSERT_DOUBLE(-0.049164429956390726, new_derivs[1], 0.01, "test calculate dc/dinput");


    teardown_backprop();

    return NULL;
}

char *test_calculate_weight_derivs(void) {
    char *error_message = NULL;
    double values[6] = {0};
    double bias[3] = {0};
    double derivs[3] = {1., 2., 3.};
    double prev_output_values[2] = {3., 2.};
    Layer layer = {.size = 3, .incoming_weights = {.width = 2, .height = 3, .values = values}, .biases = bias, .func = SIGMOID, .fan_in = 2, .fan_out = 4};

    double new_weight_values[6] = {0};
    Matrix new_weights = {.width = 2, .height = 3, .values = new_weight_values};

    calculate_weight_derivs(&layer, prev_output_values, derivs, &new_weights );

    ASSERT_DOUBLE(3., new_weight_values[0], 0.01, "test calculate weight derivs");
    ASSERT_DOUBLE(2., new_weight_values[1], 0.01, "test calculate weight derivs");
    
    ASSERT_DOUBLE(6., new_weight_values[2], 0.01, "test calculate weight derivs");
    ASSERT_DOUBLE(4., new_weight_values[3], 0.01, "test calculate weight derivs");

    ASSERT_DOUBLE(9., new_weight_values[4], 0.01, "test calculate weight derivs");
    ASSERT_DOUBLE(6., new_weight_values[5], 0.01, "test calculate weight derivs");

    return NULL;
}

char *test_derivative_of_sigmoid(void) {
    double input_values[3] = {-1.2, -0.5, 3};
    double derivs[3];

    derivative_of(SIGMOID, input_values, 3, derivs);

    char *error_message = NULL;
    ASSERT_DOUBLE(0.1778944406468057, derivs[0], 0.01, "test derivative of sigmoid");
    ASSERT_DOUBLE(0.2350037122015945, derivs[1], 0.01, "test derivative of sigmoid");
    ASSERT_DOUBLE(0.045176659730912144, derivs[2], 0.01, "test derivative of sigmoid");

    return NULL;
}

char *test_derivative_of_relu(void) {
    double input_values[3] = {-1.2, -0.5, 3};
    double derivs[3];

    derivative_of(RELU, input_values, 3, derivs);

    char *error_message = NULL;
    ASSERT_DOUBLE(0, derivs[0], 0.01, "test derivative of relu");
    ASSERT_DOUBLE(0, derivs[1], 0.01, "test derivative of relu");
    ASSERT_DOUBLE(1, derivs[2], 0.01, "test derivative of relu");

    return NULL;

}


char *test_calculate_sigmoid_primes(void) {
    double input_values[3] = {-1.2, -0.5, 3};
    double derivs[3];

    calculate_sigmoid_primes(input_values, derivs, 3);

    char *error_message = NULL;
    ASSERT_DOUBLE(0.1778944406468057, derivs[0], 0.01, "test calculate sigmoid primes");
    ASSERT_DOUBLE(0.2350037122015945, derivs[1], 0.01, "test calculate sigmoid primes");
    ASSERT_DOUBLE(0.045176659730912144, derivs[2], 0.01, "test calculate sigmoid primes");

    return NULL;
}

char *test_calculate_relu_primes(void) {
    double input_values[3] = {-1, 0, 1};
    double derivs[3];

    calculate_relu_primes(input_values, derivs, 3);

    char *error_message = NULL;
    ASSERT_DOUBLE(0, derivs[0], 0.01, "test calculate relu primes");
    ASSERT_DOUBLE(0.5, derivs[1], 0.01, "test calculate relu primes");
    ASSERT_DOUBLE(1, derivs[2], 0.01, "test calculate relu primes");

    return NULL;
}

char *test_calculate_softmax_prime(void) {
    initialize_backprop(3);

    double input_values[3] = {0.2222222222222222, 0.333333333333333, 0.4444444444444444};
    double real_values[3] = {0.33333333, 0.33333333, 0.3333333};
    double derivs[3];

    calculate_softmax_prime(input_values, real_values, derivs, 3);

    char *error_message = NULL;
    ASSERT_DOUBLE(-0.06035666, derivs[0], 0.01, "calculate softmax prime");
    ASSERT_DOUBLE(-0.01646091, derivs[1], 0.01, "calculate softmax prime");
    ASSERT_DOUBLE(0.07681757, derivs[2], 0.01, "calculate softmax prime");

    teardown_backprop();

    return NULL;
}

char *test_update_network(void) {
    char *error_message = NULL;

    Layer_Info_t layers[2] = {
        {.size = 2, .activation = SIGMOID},
        {.size = 3, .activation = SOFTMAX}
    };

    Network *network = build_network(3, layers, 2);
    if (network == NULL) {
        return NULL;
    }

    double *biases_1 = network->layers_array[1].biases;
    double *biases_2 = network->layers_array[2].biases;

    biases_1[0] = 1;
    biases_1[1] = 2;

    biases_2[0] = -1;
    biases_2[1] = -1.5;
    biases_2[2] = -2.3;

    double * weights_1 = network->layers_array[1].incoming_weights.values;

    for (int i = 0; i < 6; i++) {
        weights_1[i] = 1<<i;
    }

    double *weights_2 = network->layers_array[2].incoming_weights.values;

    for (int i = 0; i < 6; i++) {
        weights_2[i] = 6 - i;
    }

    double bias_derivs_1[2] = {2., 3.};
    double bias_derivs_2[3] = {1.4, 2, 3};

    double values_1[6] = {0};
    double values_2[6] = {0};
    Matrix deriv_weights_1 = {.height = 2, .width = 3, .values = values_1};
    Matrix deriv_weights_2 = {.height = 3, .width = 2, .values = values_2};
    Backprop_Output_t prop_outputs[2] = {
        {.weight_derivs = &deriv_weights_1, .bias_derivs = bias_derivs_1},
        {.weight_derivs = &deriv_weights_2, .bias_derivs = bias_derivs_2}
    };

    for (int i = 0; i < 6; i++) {
        prop_outputs[0].weight_derivs->values[i] = 1<<i;
        prop_outputs[1].weight_derivs->values[i] = 6 - i;
    }
    
    update_network(network, prop_outputs);

    ASSERT_DOUBLE(2, weights_1[0], 0.01, "test update network");
    ASSERT_DOUBLE(4, weights_1[1], 0.01, "test update network");
    ASSERT_DOUBLE(8, weights_1[2], 0.01, "test update network");
    ASSERT_DOUBLE(16, weights_1[3], 0.01, "test update network");
    ASSERT_DOUBLE(32, weights_1[4], 0.01, "test update network");
    ASSERT_DOUBLE(64, weights_1[5], 0.01, "test update network");

    ASSERT_DOUBLE(3, biases_1[0], 0.01, "test update network");
    ASSERT_DOUBLE(5, biases_1[1], 0.01, "test update network");

    ASSERT_DOUBLE(12, weights_2[0], 0.01, "test update network");
    ASSERT_DOUBLE(10, weights_2[1], 0.01, "test update network");
    ASSERT_DOUBLE(8, weights_2[2], 0.01, "test update network");
    ASSERT_DOUBLE(6, weights_2[3], 0.01, "test update network");
    ASSERT_DOUBLE(4, weights_2[4], 0.01, "test update network");
    ASSERT_DOUBLE(2, weights_2[5], 0.01, "test update network");

    ASSERT_DOUBLE(0.4, biases_2[0], 0.01, "test update network");
    ASSERT_DOUBLE(0.5, biases_2[1], 0.01, "test update network");
    ASSERT_DOUBLE(0.7, biases_2[2], 0.01, "test update network");

    destroy_network(network);
    return NULL;
}
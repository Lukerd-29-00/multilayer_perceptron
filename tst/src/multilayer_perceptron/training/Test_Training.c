#include "../../../../src/multilayer_perceptron/training/training.h"
#include "../../../../src/multilayer_perceptron/basic/basic.h"
#include "../../Testing.h"

Network *create_test_network() {
    Layer_Info_t info1 = {.size = 2, .activation = SIGMOID};
    Layer_Info_t info2 = {.size = 2, .activation = SOFTMAX};
    
    Layer_Info_t infos[2] = {info1, info2};

    return build_network(2, infos, 2);
}

void set_params(Network *network) {
    for (int i = 0; i < 2; i++) {
        network->layers_array[1].biases[i] = i;
        network->layers_array[2].biases[i] = -i;
    }

    for (int i = 0; i < 4; i++) {
        network->layers_array[1].incoming_weights.values[i] = 0.5 * i;
        network->layers_array[2].incoming_weights.values[i] = -0.5 * i;
    }
}

char *test_feed_forward_training() {
    Network *network = create_test_network();
    char *error_message = NULL;
    if (network == NULL) {
        return NULL;
    }
    Layer_Calcs_t *calcs = initialize_calcs(network);
    if (calcs == NULL) {
        destroy_network(network);
        return NULL;
    }

    set_params(network);

    double input_values[2] = {1.5, 2};

    feed_forward_training(network, input_values, calcs);

    ASSERT_DOUBLE(1.5, calcs[0].input_values[0], 0.01, "test training feed forward");
    ASSERT_DOUBLE(2, calcs[0].input_values[1], 0.01, "test training feed forward");
    
    ASSERT_DOUBLE(1.5, calcs[0].output_values[0], 0.01, "test training feed forward");
    ASSERT_DOUBLE(2, calcs[0].output_values[1], 0.01, "test training feed forward");

    ASSERT_DOUBLE(1, calcs[1].input_values[0], 0.01, "test training feed forward");
    ASSERT_DOUBLE(5.5, calcs[1].input_values[1], 0.01, "test training feed forward");

    ASSERT_DOUBLE(0.7310585786300049, calcs[1].output_values[0], 0.01, "test training feed forward");
    ASSERT_DOUBLE(0.995929862284104, calcs[1].output_values[1], 0.01, "test training feed forward");

    ASSERT_DOUBLE(-0.49796493, calcs[2].input_values[0], 0.01, "test training feed forward");
    ASSERT_DOUBLE(-3.22495337, calcs[2].input_values[1], 0.01, "test training feed forward");

    ASSERT_DOUBLE(0.93860051, calcs[2].output_values[0], 0.01, "test training feed forward");
    ASSERT_DOUBLE(0.06139949, calcs[2].output_values[1], 0.01, "test training feed forward");

}

char *test_train_on_sample() {
    Network *network = NULL;
    Layer_Calcs_t *calcs = NULL;
    Backprop_Output_t *boutput = NULL;
    Backprop_Output_t *avg = NULL;
    char *error_message = NULL;

    network = create_test_network();
    if (network == NULL) {
        goto end;
    }
    initialize_backprop(network->largest_layer_size);
    calcs = initialize_calcs(network);
    if (calcs == NULL) {
        goto end;
    }
    boutput = initialize_backprop_outputs(network);
    if (boutput == NULL) {
        goto end;
    }
    avg = initialize_backprop_outputs(network);
    if (avg == NULL) {
        goto end;
    }

    set_params(network);
    double input_values_1[2] = {1.5, 2};
    double correct_answers_1[2] = {0.5, 0.5};

    double input_values_2[2] = {-1.5, -2};
    double correct_answers_2[2] = {0.25, 0.75};

    Training_Run_t run_1 = {.input_values = input_values_1, .correct_answer = correct_answers_1};
    Training_Run_t run_2 = {.input_values = input_values_2, .correct_answer = correct_answers_2};

    Training_Run_t runs[2] = {run_1, run_2};

    train_on_sample(network, runs, 2, calcs, boutput, avg);

    ASSERT_DOUBLE(0.02981781, avg[0].weight_derivs->values[0], 0.01, "test train on sample");
    ASSERT_DOUBLE(0.03975709, avg[0].weight_derivs->values[1], 0.01, "test train on sample");
    ASSERT_DOUBLE(0.00061476, avg[0].weight_derivs->values[2], 0.01, "test train on sample");
    ASSERT_DOUBLE(0.00081968, avg[0].weight_derivs->values[3], 0.01, "test train on sample");

    ASSERT_DOUBLE(0.0147957, avg[0].bias_derivs[0], 0.01, "test train on sample");
    ASSERT_DOUBLE(0.00117137, avg[0].bias_derivs[1], 0.01, "test train on sample");

    ASSERT_DOUBLE(0.07391402, avg[1].weight_derivs->values[0], 0.01, "test train on sample");
    ASSERT_DOUBLE(0.10069396, avg[1].weight_derivs->values[1], 0.01, "test train on sample");
    ASSERT_DOUBLE(-0.07391402, avg[1].weight_derivs->values[2], 0.01, "test train on sample");
    ASSERT_DOUBLE(-0.10069396, avg[1].weight_derivs->values[3], 0.01, "test train on sample");

    ASSERT_DOUBLE(0.05995823, avg[1].bias_derivs[0], 0.01, "test train on sample");
    ASSERT_DOUBLE(-0.0712078, avg[1].bias_derivs[1], 0.01, "test train on sample");

    end:
    if (network != NULL) {
        teardown_backprop();
        destroy_network(network);
    }
    if (calcs != NULL) {
        teardown_calcs(calcs, 3);
    }
    if (boutput != NULL) {
        teardown_backprop_outputs(boutput, 2);
    }
    if (avg != NULL) {
        teardown_backprop_outputs(avg, 2);
    }
    return NULL;

}
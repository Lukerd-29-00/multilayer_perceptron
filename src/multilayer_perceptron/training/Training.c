#include <sqlite3.h>
#include "../../retrieval/retrieval.h"
#include "../../linal/linal.h"
#include "../basic/basic.h"
#include "../../activation/activation.h"
#include "training.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

void set_backprop_output_zero(Network *network, Backprop_Output_t *output) {
    for (int i = 0; i < network->layers_count - 1; i++) {
        for (int j = 0; j < network->layers_array[i+1].size; j++) {
            output[i].bias_derivs[j] = 0;
        }

        for (int j = 0; j < network->layers_array[i+1].incoming_weights.height * network->layers_array[i+1].incoming_weights.width; j++) {
            output[i].weight_derivs->values[j] = 0;
        }
    }
    
}

void sum_to_avg(const Network *network, Backprop_Output_t *output, size_t sample_size) {
    for (int i = 0; i < network->layers_count - 1; i++) {
        scale_matrix(output[i].weight_derivs, 1./sample_size);
        scale_vec(output[i].bias_derivs, network->layers_array[i+1].size, 1./sample_size);
    }
}

Training_Run_t *initialize_training_runs(Network *network, size_t population_size, size_t sample_count) {
    size_t sample_size = (population_size / sample_count) + 1;
    Training_Run_t *output = malloc(sample_size * sizeof(Training_Run_t));
    if (output == NULL) {
        return NULL;
    }
    for (int j = 0; j < sample_size; j++) {
        output[j].input_values = NULL;
        output[j].correct_answer = NULL;
    }

    bool failed = false;
    int i;
    for (i = 0; i < sample_size; i++) {
        output[i].input_values = malloc(sizeof(double) * network->layers_array[0].size);
        if (output[i].input_values == NULL) {
            failed = true;
            break;
        }
        output[i].correct_answer = malloc(sizeof(double) * network->layers_array[network->layers_count - 1].size);
        if (output[i].correct_answer == NULL) {
            failed = true;
            break;
        }
    }
    if (failed) {
        for (int j = 0; j < i; j++) {
            if (output[j].input_values != NULL) {
                free(output[j].input_values);
            }
            if (output[j].correct_answer != NULL) {
                free(output[j].correct_answer);
            }
        }
        free(output);
        return NULL;
    }
    return output;
}

Layer_Calcs_t *initialize_calcs(Network *network) {
    Layer_Calcs_t *calcs = malloc(sizeof(Layer_Calcs_t) * network->layers_count);
    if (calcs == NULL) {
        return NULL;
    }

    for (int j = 0; j < network->layers_count; j++) {
        calcs[j].input_values = NULL;
        calcs[j].output_values = NULL;
    }

    int i;
    bool failed = false;
    for (i = 0; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        calcs[i].input_values = malloc(sizeof(double) * layer.size);
        if (calcs[i].input_values == NULL) {
            failed = true;
            break;
        }
        calcs[i].output_values = malloc(sizeof(double) * layer.size);
        if (calcs[i].output_values == NULL) {
            failed = true;
            break;
        }
    }

    if (failed) {
        for (int j = 0; j < i; j++) {
            if (calcs[j].input_values != NULL) {
                free(calcs[j].input_values);
            }
            if (calcs[j].output_values != NULL) {
                free(calcs[j].output_values);
            }
        }
        free(calcs);
        return NULL;
    }
    return calcs;
}

void teardown_training_runs(Training_Run_t *training_runs, size_t population_size, size_t sample_count) {
    int sample_size = (population_size / sample_count) + 1;
    for (int i = 0; i < sample_size; i++) {
        free(training_runs[i].input_values);
        free(training_runs[i].correct_answer);
    }
    free(training_runs);
}

void teardown_calcs(Layer_Calcs_t *calcs, size_t size) {
    for (int i = 0; i < size; i++) {
        free(calcs[i].input_values);
        free(calcs[i].output_values);
    }
    free(calcs);
}

int sample_boundary(const int pop_size, const int sample_count, const int idx) {
    return ((idx % sample_count) * pop_size) / sample_count;
}

void apply_activation_training(Activation func, const double * restrict transformed_vec, const size_t vector_size, double * restrict output) {
    switch (func) {
        case SIGMOID:
            for (int i = 0; i < vector_size; i++) {
                 output[i] = sigmoid(transformed_vec[i]);
            }
        break;
        case RELU:
            for (int i = 0; i < vector_size; i++) {
                output[i] = relu(transformed_vec[i]);
            }
        break;
        case SOFTMAX:
            softmax(transformed_vec, vector_size, output);
            break;
        case NONE:
            errno = EINVAL;
            break;
    }
}

void feed_forward_training(const Network *network, double *input_values, Layer_Calcs_t *calcs) {
    size_t first_layer_size = network->layers_array[0].size;
    memcpy(calcs[0].input_values, input_values, sizeof(double) * first_layer_size);
    memcpy(calcs[0].output_values, input_values, sizeof(double) * first_layer_size);
    for (int i = 1; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        transform(&layer.incoming_weights, calcs[i-1].output_values, calcs[i].input_values);
        add(calcs[i].input_values, layer.biases, layer.size);
        apply_activation_training(layer.func, calcs[i].input_values, layer.size, calcs[i].output_values);
    }
}

void train_on_sample(const Network *network, const Training_Run_t *sample, const size_t sample_size, Layer_Calcs_t *calcs, Backprop_Output_t *backprop_output, Backprop_Output_t *avg_backprop) {
    for (int i = 0; i < sample_size; i++) {
        feed_forward_training(network, sample[i].input_values, calcs);
        calculate_backprop_for_run(network, calcs, sample[i].correct_answer, backprop_output);
        for (int j = 1; j < network->layers_count; j++) {
            add_matrices(avg_backprop[j-1].weight_derivs, backprop_output[j-1].weight_derivs);
            add(avg_backprop[j-1].bias_derivs, backprop_output[j-1].bias_derivs, network->layers_array[j].size);
        }
    }
    sum_to_avg(network, avg_backprop, sample_size);
}

void train(Network *network, const char *data_file, const char *learnset, const int sample_count, const int runs) {
    Training_Run_t *sample = NULL;
    Layer_Calcs_t *calcs = NULL;
    Backprop_Output_t *outputs = NULL;
    Backprop_Output_t *avg = NULL;
    sqlite3 *conn = NULL;

    initialize_backprop(network->largest_layer_size);
    if (errno) {
        goto end;
    }

    int ok = sqlite3_open(data_file, &conn);
    if (ok != SQLITE_OK) {
        errno = FAILED_CONNECTION;
        conn = NULL;
        goto end;
    }

    initialize_retrieval(conn);
    if (errno) {
        goto end;
    }

    int population_size = learnset_size(learnset);
    sample = initialize_training_runs(network, population_size, sample_count);
    if (sample == NULL) {
        goto end;
    }
    calcs = initialize_calcs(network);
    if (calcs == NULL) {
        goto end;
    }
    outputs = initialize_backprop_outputs(network);
    if (outputs == NULL) {
        goto end;
    }
    avg = initialize_backprop_outputs(network);
    if (avg == NULL) {
        goto end;
    }
    int idx = 0;
    for (int i = 0; i < runs; i++) {
        set_backprop_output_zero(network, avg);
        int current_boundary = sample_boundary(population_size, sample_count, idx);
        int next_boundary = sample_boundary(population_size, sample_count, idx + 1);
        if (next_boundary == 0) {
            next_boundary = population_size;
        }
        int this_sample_size = next_boundary - current_boundary;
        load(learnset, current_boundary, this_sample_size, sample);
        if (errno) {
            break;
        }
        train_on_sample(network, sample, this_sample_size, calcs, outputs, avg);
        update_network(network, avg);
    }

    end:
    if (outputs != NULL) {
        teardown_backprop_outputs(outputs, network->layers_count - 1);
    }
    if (calcs != NULL) {
        teardown_calcs(calcs, network->layers_count);
    }
    if (sample != NULL) {
        teardown_training_runs(sample, population_size, sample_count);
    }
    if (avg != NULL) {
        teardown_backprop_outputs(avg, network->layers_count - 1);
    }
    stop_retrieval();
    teardown_backprop();
    if (conn != NULL) {
        sqlite3_close(conn);
    }
}
#include "../basic/basic.h"
#include "../training/training.h"
#include "../../retrieval/retrieval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void print_NN_output(Network *network, double *input) {
    double state_vec_1[6];
    double state_vec_2[6];
    double *output = feed_forward(network, input, state_vec_1, state_vec_2);

    printf("[%.4f, %.4f] -> [%.4f, %.4f]\n", input[0], input[1], output[0], output[1]);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }

    char *training_data_file = argv[1];

    Layer_Info_t info_3 = {.size = 4, .activation = RELU};
    Layer_Info_t info_4 = {.size = 6, .activation = SIGMOID};
    Layer_Info_t info_5 = {.size = 2, .activation = SOFTMAX};

    Layer_Info_t infos[3] = {info_3, info_4, info_5};

    Network *network = build_network(2, infos, 3);
    if (network == NULL) {
        perror("Could not build network: ");
        return 2;
    }
    initialize_for_training(network);


    double input[2] = {0, 0};
    print_NN_output(network,input);

    input[0] = 0;
    input[1] = 1;
    print_NN_output(network, input);

    input[0] = 1;
    input[1] = 0;
    print_NN_output(network, input);

    input[0] = 1;
    input[1] = 1;
    print_NN_output(network, input);

    train(network, training_data_file, "AND_GATE", 1, 10000);
    if (errno) {
        perror("Error during training: ");
        return 3;
    }

    input[0] = input[1] = 0;
    print_NN_output(network, input);

    input[0] = 0;
    input[1] = 1;
    print_NN_output(network, input);

    input[0] = 1;
    input[1] = 0;
    print_NN_output(network, input);

    input[0] = 1;
    input[1] = 1;
    print_NN_output(network, input);

    return 0;

}
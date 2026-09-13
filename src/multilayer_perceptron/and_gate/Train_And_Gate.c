#include "../basic/basic.h"
#include "../training/training.h"
#include "../../retrieval/retrieval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void print_NN_output(Network *network, double *input) {
    double state_vec_1[4];
    double state_vec_2[4];
    double *output = feed_forward(network, input, state_vec_1, state_vec_2);

    printf("[%.2f, %.2f] -> [%.2f, %.2f]\n", input[0], input[1], output[0], output[1]);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }

    char *training_data_file = argv[1];

    Layer_Info_t info_1 = {.size = 4, .activation = SIGMOID};
    Layer_Info_t info_2 = {.size = 4, .activation = RELU};
    Layer_Info_t info_3 = {.size = 4, .activation = RELU};
    Layer_Info_t info_4 = {.size = 4, .activation = SIGMOID};
    Layer_Info_t info_5 = {.size = 2, .activation = SOFTMAX};

    Layer_Info_t infos[5] = {info_1, info_2, info_3, info_4, info_5};

    Network *network = build_network(2, infos, 5);
    if (network == NULL) {
        perror("Could not build network: ");
        return 2;
    }

    train(network, training_data_file, "AND_GATE", 1, 100);
    if (errno) {
        perror("Error during training: ");
        return 3;
    }

    double input[2] = {0., 0.};
    print_NN_output(network, input);

    input[0] = 1;
    print_NN_output(network, input);

    input[0] = 0;
    input[1] = 1;
    print_NN_output(network, input);

    input[0] = 1;
    print_NN_output(network, input);

    return 0;

}
#include "basic.h"
#include "../../linal/linal.h"
#include <stddef.h>
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

Network *build_network(size_t input_size, Layer_Info_t *layer_info, size_t active_layer_count) {
    
    assert(active_layer_count >= 1);
    size_t total_layer_count = active_layer_count + 1;
    Network *output = malloc(sizeof(Network));
    if (output == NULL) {
        return NULL;
    }
    output->layers_array = malloc(sizeof(Layer) * total_layer_count);
    if (output->layers_array == NULL) {
        free(output);
        return NULL;
    }
    output->layers_array[0] = initialize_first_layer(input_size, layer_info[0].size);
    size_t max_size = output->layers_array[0].size;
    for (int i = 0; i < active_layer_count; i++) {
        Layer_Info_t info = layer_info[i];
        size_t fan_in = i == 0 ? output->layers_array[0].size : layer_info[i-1].size;
        size_t fan_out = i == active_layer_count - 1 ? NO_FAN : layer_info[i + 1].size;
        
        Layer current = initialize_layer(info.size, info.activation, fan_in, fan_out);

        if (errno) {
            destroy_network(output);
            return NULL;
        }
        output->layers_array[i + 1] = current;
        
        if (current.size > max_size) {
            max_size = current.size;
        }
        
    }
    
    output->layers_count = total_layer_count;
    output->largest_layer_size = max_size;
    return output;
}

double *feed_forward(const Network * const network, double * const input_vec, double * restrict state_vec_1, double * restrict state_vec_2) {
    
    for (int i = 0; i < network->layers_array[0].size; i++) {
        state_vec_1[i] = input_vec[i];
    }
    
    for (int i = 1; i < network->layers_count; i++) {
        Layer layer = network->layers_array[i];
        apply_transformation(layer, state_vec_1, state_vec_2);
        apply_activation(layer.func, state_vec_2, layer.size);
        
        double *tmp = state_vec_1;
        state_vec_1 = state_vec_2;
        state_vec_2 = tmp;
    }
    return state_vec_1;
}


void destroy_network(Network *network) {
    for(int i = 0; i < network->layers_count; i++) {
        destroy_layer(network->layers_array[i]);
    }
    free(network->layers_array);
    free(network);
}

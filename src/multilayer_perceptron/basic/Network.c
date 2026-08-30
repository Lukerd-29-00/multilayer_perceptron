#include "basic.h"
#include "..\..\linal\linal.h"
#include <stddef.h>
#include <errno.h>
#include <stdlib.h>

Layer *build_network(size_t input_size, Layer_Info_t *layer_info, size_t active_layer_count) {
    Layer *initial_layer = initialize_first_layer(input_size);
    if (initial_layer == NULL) {
        return NULL;
    }
    Layer *prev = initial_layer;
    for (int i = 0; i < active_layer_count; i++) {
        Layer_Info_t info = layer_info[i];
        Layer * current = initialize_layer(info.size, info.activation, prev->size);
        if (current == NULL) {
            destroy_network(initial_layer);
            return NULL;
        }
        prev->next = current;
        current->prev = prev;
        prev = current;
    }
    return initial_layer;
}

double *feed_forward(const Layer * const network, double * const input_vec, double * restrict state_vec_1, double * restrict state_vec_2) {
    
    for (int i = 0; i < network->size; i++) {
        state_vec_1[i] = input_vec[i];
    }
    
    for (Layer *ptr = network->next; ptr != NULL; ptr = ptr->next) {
        apply_transformation(ptr, state_vec_1, state_vec_2);
        apply_activation(ptr->func, state_vec_2, ptr->size);
        
        double *tmp = state_vec_1;
        state_vec_1 = state_vec_2;
        state_vec_2 = tmp;
    }
    return state_vec_1;
}


void destroy_network(Layer *first) {
    Layer *nxt;
	for (Layer *ptr = first; ptr != NULL; ptr = nxt) {
        nxt = ptr->next;
        destroy_layer(ptr);
    }
}

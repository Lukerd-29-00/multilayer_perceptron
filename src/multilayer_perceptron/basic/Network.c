#include "basic.h"
#include "..\..\linal\linal.h"
#include <stddef.h>
#include <errno.h>
#include <stdlib.h>

Layer *build_network(size_t input_size, Layer_Info_t *layer_info, size_t active_layer_count) {
    Layer *initial_layer = initialize_layer(input_size, NONE, NO_INCOMING_WEIGHTS);
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

//Needs optimizing: should pass through a single double * and not do any heap allocation
double *feed_forward(const Layer * const network, double * const input_vec) {
    if (network->next == NULL) {
        errno = EINVAL;
        return NULL;
    }
    double *current_state = malloc(sizeof(double) * network->size);
    for (int i = 0; i < network->size; i++) 
        current_state[i] = input_vec[i];
    for (Layer *ptr = network->next; ptr != NULL; ptr = ptr->next) {
        double *transformed_vec = apply_transformation(ptr, current_state);
        apply_activation(ptr->func, transformed_vec, ptr->size);
        free(current_state);
        current_state = transformed_vec;
    }
    return current_state;
}


void destroy_network(Layer *first) {
    Layer *nxt;
	for (Layer *ptr = first; ptr != NULL; ptr = nxt) {
        nxt = ptr->next;
        destroy_layer(ptr);
    }
}

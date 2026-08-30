#include "basic.h"
#include "..\..\linal\linal.h"
#include <stddef.h>
#include <errno.h>

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
Vector *feed_forward(const Layer * const network, Vector *input_vec) {
    Layer * start = network->next;
    if (start == NULL) {
        errno = EINVAL;
        return NULL;
    }
    Vector *current_state = apply_transformation(start, input_vec);
    apply_activation_destructive(start->func, current_state);
    for (Layer *ptr = start; ptr != NULL; ptr = ptr->next) {
        Vector *transformed_vec = apply_transformation(ptr, current_state);
        apply_activation_destructive(ptr->func, transformed_vec);
        destroy_vector(current_state);
        current_state = transformed_vec;
    }
    return current_state;
}


void destroy_network(Layer *first) {
    Layer *nxt = first;
    for (Layer *ptr = first; ptr != NULL; ptr = nxt) {
        nxt = ptr->next;
        destroy_layer(ptr);
    }
}
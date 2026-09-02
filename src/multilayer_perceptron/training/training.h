#include "../basic/basic.h"

double glorot(size_t fan_in, size_t fan_out);
double he(size_t fan_in);
void initialize_layer_glorot(Layer layer);
void initialize_layer_he(Layer layer);
void initialize_for_training(Network *network);
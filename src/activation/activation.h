double relu(double x);
double relu_prime(double x);
double sigmoid(double x);
double sigmoid_prime(double x);
void softmax(double * restrict all_values, int num_values, double * restrict output);
double softmax_prime(double *input_values, int idx, int num_values);
void softmax_destructive(double *all_values, int num_values);
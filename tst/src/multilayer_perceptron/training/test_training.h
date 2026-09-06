
void training_suite(int *failures);
char *test_glorot(void);
char *test_he(void);
char *test_glorot_initialization(void);
char *test_he_initialization(void);
char *test_initialize_for_training(void);

char *test_calculate_backprop_for_run(void);
char *test_calculate_dc_dinput_hidden(void);
char *test_calculate_weight_derivs(void);
char *test_derivative_of_sigmoid(void);
char *test_derivative_of_relu(void);
char *test_calculate_sigmoid_primes(void);
char *test_calculate_relu_primes(void);
char *test_calculate_softmax_prime(void);
char *test_update_network(void);

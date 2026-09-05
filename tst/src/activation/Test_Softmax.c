#include "../Testing.h"
#include "../../../src/activation/activation.h"

#define ERROR_MARGIN 0.01

char *test_softmax() {
    double input_vec[4] = {12, 15, 13, 11};
    double output_vec[4];

    softmax(input_vec, 4, output_vec);

    char *error_message = NULL;

    ASSERT_DOUBLE(0.04137069692096015, output_vec[0], ERROR_MARGIN, "softmax");
    ASSERT_DOUBLE(0.8309526605439514, output_vec[1], ERROR_MARGIN, "softmax");
    ASSERT_DOUBLE(0.11245721367093255, output_vec[2], ERROR_MARGIN, "softmax");
    ASSERT_DOUBLE(0.015219428864155927, output_vec[3], ERROR_MARGIN, "softmax");
    return NULL;
}

char *test_softmax_destructive() {
    double vec[4] = {12, 15, 13, 11};

    softmax_destructive(vec, 4);

    char *error_message = NULL;

    ASSERT_DOUBLE(0.04137069692096015, vec[0], ERROR_MARGIN, "softmax");
    ASSERT_DOUBLE(0.8309526605439514, vec[1], ERROR_MARGIN, "softmax");
    ASSERT_DOUBLE(0.11245721367093255, vec[2], ERROR_MARGIN, "softmax");
    ASSERT_DOUBLE(0.015219428864155927, vec[3], ERROR_MARGIN, "softmax");
    return NULL;
}

char *test_softmax_prime() {
    double input_vec[3] = {0.25, 0.5, 0.25};
    Matrix *output_matrix = initialize_matrix(3, 3);

    softmax_prime(input_vec, output_matrix);
    double *values = output_matrix->values;
    
    char *error_message = NULL;

    ASSERT_DOUBLE(0.1875, values[0], ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(-0.125, values[1], ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(-0.0625, values[2], ERROR_MARGIN, "softmax prime");
    
    ASSERT_DOUBLE(-0.125, values[3], ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(0.25, values[4], ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(-0.125, values[5], ERROR_MARGIN, "softmax prime");

    ASSERT_DOUBLE(-0.0625, values[6], ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(-0.125, values[7], ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(0.1875, values[8], ERROR_MARGIN, "softmax prime");
    return NULL;
}
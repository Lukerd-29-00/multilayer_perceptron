#include "../Testing.h"
#include "..\..\..\src\activation\activation.h"

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
    double input_vec[4] = {-1, -1.4, -1.01, -0.97};
    
    char *error_message = NULL;

    ASSERT_DOUBLE(0.19753251352601292, softmax_prime(input_vec, 0, 4), ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(0.14863288441989186, softmax_prime(input_vec, 1, 4), ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(0.19629020254516497, softmax_prime(input_vec, 2, 4), ERROR_MARGIN, "softmax prime");
    ASSERT_DOUBLE(0.2012445311960707, softmax_prime(input_vec, 3, 4), ERROR_MARGIN, "softmax prime");
    return NULL;
}
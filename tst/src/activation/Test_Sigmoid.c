#include "../Testing.h"
#include "..\..\..\src\activation\activation.h"
#include <stdlib.h>
#include <stdio.h>

char *test_sigmoid(void) {
    double output_1 = sigmoid(10);
    double output_2 = sigmoid(3);
    double output_3 = sigmoid(-1);
    double output_4 = sigmoid(0);

    char *error_message = assert_double_approx(0.9999546021312976, output_1, 0.005);
    if (error_message != NULL) {
        return enrich_error_message(error_message, "sigmoid");
    }

    error_message = assert_double_approx(0.9525741268224334, output_2, 0.005);
    if (error_message != NULL) {
        return enrich_error_message(error_message, "sigmoid");
    }
    error_message = assert_double_approx(0.2689414213699951, output_3, 0.005);
    if (error_message != NULL) {
        return enrich_error_message(error_message, "sigmoid");
    }
    error_message = assert_double_approx(0.5, output_4, 0.005);
    if (error_message != NULL) {
        return enrich_error_message(error_message, "sigmoid");
    }
    return NULL;
}

char *test_sigmoid_prime(void) {
    double output = sigmoid_prime(-1.5);

    char *error_message = assert_double_approx(0.14914645207033286, output, 0.005);
    if (error_message != NULL) {
        return enrich_error_message(error_message, "sigmoid prime");
    }
    return NULL;
}
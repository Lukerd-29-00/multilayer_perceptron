#include "../../../src/linal/linal.h"
#include "../Testing.h"
#include "linal_test.h"
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <windows.h>

char *test_vector_add(void) {
    double a[3];
    double b[3];
    
    a[0] = 1.;
    a[1] = 2.;
    a[2] = 3.;

    b[0] = -4.;
    b[1] = 3.;
    b[2] = -3.;

    add(a, b, 3);
    char *error_format = NULL;
    char *error_message = assert_double_approx(-3., a[0], 0.01);

    if (error_message != NULL) {
        error_format = "Error in test_vector_add x coordinate: %s\n";
        goto end;
    } else if (errno) {
        goto end;
    }

    error_message = assert_double_approx(5., a[1], 0.01);
    if (error_message != NULL) {
        error_format = "Error in test_vector_add y coordinate: %s\n";
        goto end;
    } else if (errno) {
        goto end;
    }

    error_message = assert_double_approx(0., a[2], 0.01);
    if (error_message != NULL) {
        error_format = "Error in test_vector_add z coordinate: %s\n";
        goto end;
    } else if (errno) {
        goto end;
    }

    end:
        if (error_format != NULL) {
            char buf[MAX_LEN] = {};
            snprintf(buf, MAX_LEN, error_format, error_message);
            char *rich_message = build_error_message(buf);
            free(error_message);
            return rich_message;
        }
        return NULL;
}

char *test_transform(double *vec, Matrix *mat, double *expected_output) {
    double * const output = malloc(sizeof(double) * mat->height);
    transform(mat, vec, output);
    char *error_message = NULL;

    for (int i = 0; i < mat->height; i++) {
    	ASSERT_DOUBLE(expected_output[i], output[i], 0.01, "transform test");
    }
    
    free(output);
    return NULL;
}

char *test_transform_3x3(void) {
    double input_vector[3] = {2., 3., 4.};
    double expected_output[3] = {23., 45., 68.};
    Matrix *transformation = initialize_matrix(3, 3);
    if (transformation == NULL) {
        goto end;
    }
    transformation->values[0] = 1;
    transformation->values[1] = 3;
    transformation->values[2] = 3;

    transformation->values[3] = 5;
    transformation->values[4] = 5;
    transformation->values[5] = 5;

    transformation->values[6] = 8;
    transformation->values[7] = 8;
    transformation->values[8] = 7;

    char *result = test_transform(input_vector, transformation, expected_output);
    char *test_result = NULL;
    if (result != NULL) {
        char buf[MAX_LEN];
        strcpy(buf, "Error in test transform 3x3: ");
        strcat(buf, result);
        free(result);
        test_result = build_error_message(buf);
    }

    end:

    if (transformation != NULL) {
        destroy_matrix(transformation);
    }
    return test_result;
}

char *test_transform_3x2(void) {
    double input_vector[2] = {3.5, -8.};
    double expected_output[3] = {11.5, -24., -34.};
    Matrix *transformation = initialize_matrix(3, 2);
    
    transformation->values[0] = 1;
    transformation->values[1] = -1;

    transformation->values[2] = 0;
    transformation->values[3] = 3;

    transformation->values[4] = 4;
    transformation->values[5] = 6;

    char *result = test_transform(input_vector, transformation, expected_output);
    char *test_result = NULL;
    if (result != NULL) {
        char buf[MAX_LEN];
        strcpy(buf, "Error in test transform 3x2: ");
        strcat(buf, result);
        free(result);
        test_result = build_error_message(buf);
    }

    if (transformation != NULL) {
        destroy_matrix(transformation);
    }
    return test_result;
}

char *test_transform_2x3(void) {
    double input_vector[3] = {2., 3., 4.};
    double expected_output[2] = {-21.59, -10.};
    Matrix *transformation = initialize_matrix(2, 3);

    transformation->values[0] = 1;
    transformation->values[1] = -1;

    transformation->values[2] = 0;
    transformation->values[3] = 3;

    transformation->values[4] = 4;
    transformation->values[5] = 6;

    char *result = test_transform(input_vector, transformation, expected_output);
    char *test_result = NULL;
    if (result != NULL) {
        char buf[MAX_LEN];
        strcpy(buf, "Error in test transform 2x3: ");
        strcat(buf, result);
        free(result);
        test_result = build_error_message(buf);
    }

    if (transformation != NULL) {
        destroy_matrix(transformation);
    }
    return test_result;
}

char *test_outer_product(void) {
    char *error_message = NULL;

    double vector_a[4] = {2., 3.4, 5., -3.};
    double vector_b[3] = {2., 3.33, 8.};

    Matrix *output_matrix = initialize_matrix(4, 3);
    if (output_matrix == NULL) {
        return NULL;
    }

    outer_product(vector_a, 4, vector_b, 3, output_matrix);

    ASSERT_DOUBLE(4, output_matrix->values[0], 0.01, "test outer product");
    ASSERT_DOUBLE(6.66, output_matrix->values[1], 0.01, "test outer product");
    ASSERT_DOUBLE(16, output_matrix->values[2], 0.01, "test outer product");

    ASSERT_DOUBLE(6.8, output_matrix->values[3], 0.01, "test outer product");
    ASSERT_DOUBLE(11.322, output_matrix->values[4], 0.01, "test outer product");
    ASSERT_DOUBLE(27.2, output_matrix->values[5], 0.01, "test outer product");

    ASSERT_DOUBLE(10, output_matrix->values[6], 0.01, "test outer product");
    ASSERT_DOUBLE(16.65, output_matrix->values[7], 0.01, "test outer product");
    ASSERT_DOUBLE(40, output_matrix->values[8], 0.01, "test outer product");

    ASSERT_DOUBLE(-6, output_matrix->values[9], 0.01, "test outer product");
    ASSERT_DOUBLE(-9.99, output_matrix->values[10], 0.01, "test outer product");
    ASSERT_DOUBLE(-24, output_matrix->values[11], 0.01, "test outer product");

    return NULL;
}

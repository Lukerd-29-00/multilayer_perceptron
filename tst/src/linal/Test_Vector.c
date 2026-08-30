#include "../../../src/linal/linal.h"
#include "../Testing.h"
#include "linal_test.h"
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <windows.h>

#define MAX_LINE_LENGTH 10

const char linal_resources[16] = "linal";

int read_next_int(FILE *resource, int *target) {
    char line_buffer[MAX_LINE_LENGTH];
    int error_code = read_line(line_buffer, resource, MAX_LINE_LENGTH);
    if (error_code == -2) {
        return -2;
    }   
    else if (error_code == -1 && line_buffer[0] == '\0') {
        return -1;
    } else if (error_code == -1) {
        error_code = sscanf(line_buffer, "%d", target);
        if (error_code != 1) { 
            errno = 449;
            return -2;
        }
        return 0;
    } 
    else {
        error_code = sscanf(line_buffer, "%d\n", target);
        if (error_code != 1) {
            errno = 449;
            return -2;
        }
        return 0;
    }
}

int read_next_double(FILE *resource, double *target) {
    char line_buffer[MAX_LINE_LENGTH];
    int error_code = read_line(line_buffer, resource, MAX_LINE_LENGTH);
    if (error_code == -2) {
        return -2;
    }   
    else if (error_code == -1 && line_buffer[0] == '\0') {
        return -1;
    } else if (error_code == -1) {
        error_code = sscanf(line_buffer, "%lf", target);
        if (error_code != 1) {
            errno = 449;
            return -2;
        }
        return 0;
    } 
    else {
        error_code = sscanf(line_buffer, "%lf\n", target);
        if (error_code != 1) {
            errno = 449;
            return -2;
        }
        return 0;
    }
}

Matrix *load_mat_from_file(char *file_name) {
    char path[MAX_PATH];
    Matrix *output = NULL;
    FILE *resource = NULL;

    strcpy(path, linal_resources);
    
    if (join_path(path, file_name) == -1) {
        goto error;
    }

    resource = load_resource_file(path);
    if (resource == NULL){
        goto error;
    }

    int height;
    int error = read_next_int(resource, &height);
    if (error == -2) {
        goto error;
    } else if (error == -1) {
        errno = 450;
        goto error;
    }
    int width;
    error = read_next_int(resource, &width);
    if (error == -2) {
        goto error;
    } else if (error == -1) {
        errno = 450;
        goto error;
    }
    output = initialize_matrix(height, width);

    for (int i = 0; i < width * height; i++) {
        double n;
        int error_code = read_next_double(resource, &n);
        if (error_code == -2) {
            goto error;
        } else if (error_code == -1) {
            errno = 450;
            goto error;
        } else {
            output->values[i] = n;
        }
    }

    fclose(resource);
    return output;
    error:
    if (resource != NULL) {
        fclose(resource);
    }
    if (output != NULL) {
        destroy_matrix(output);
    }
    return NULL;
}

double *load_vec_from_file(char *file_name) {
    char path[MAX_PATH];
    double *output = NULL;
    FILE *resource = NULL;

    strcpy(path, linal_resources);
    
    if (join_path(path, file_name) == -1) {
        goto error;
    }

    resource = load_resource_file(path);
    if (resource == NULL){
        goto error;
    }

    int dimensions;
    int error = read_next_int(resource, &dimensions);
    if (error == -2) {
        goto error;
    } else if(error == -1) {
        errno = 450;
        goto error;
    }

    output = malloc(sizeof(double) * dimensions);
    if (output == NULL) {
    	goto error;
    }

    for (int i = 0;i < dimensions;i++) {
        double n;
        int error_code = read_next_double(resource, &n);
        if (error_code == -2) {
            goto error;
        } else if (error_code == -1) {
            errno = 450;
            goto error;
        } else {
            output[i] = n;
        }
    }

    fclose(resource);
    return output;
    error:
    if (resource != NULL) {
        fclose(resource);
    }
    if (output != NULL) {
        free(output);
    }
    return NULL;
}

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
    double * input_vector = NULL;
    double *expected_output = NULL;
    Matrix *transformation = NULL;
    if ((input_vector = load_vec_from_file("transform_me.txt")) == NULL) 
        goto error;
   
    if ((expected_output = load_vec_from_file("3x3/expected_output_3x3.txt")) == NULL)
        goto error;

    if ((transformation = load_mat_from_file("3x3/test_mat_3x3.txt")) == NULL)
        goto error;

    char *result = test_transform(input_vector, transformation, expected_output);
    char *test_result = NULL;
    if (result != NULL) {
        char buf[MAX_LEN];
        strcpy(buf, "Error in test transform 3x3: ");
        strcat(buf, result);
        free(result);
        test_result = build_error_message(buf);
    }

    free(input_vector);
    free(expected_output);
    destroy_matrix(transformation);
    return test_result;

    error:
    if (input_vector != NULL) {
        free(input_vector);
    }
    if (expected_output != NULL) {
        free(expected_output);
    }
    if (transformation != NULL) {
        destroy_matrix(transformation);
    }
    return NULL;

}

char *test_transform_3x2(void) {
    double * input_vector = NULL;
    double *expected_output = NULL;
    Matrix *transformation = NULL;
    if ((input_vector = load_vec_from_file("transform_me_2d.txt")) == NULL) 
        goto error;
   
    if ((expected_output = load_vec_from_file("3x2/expected_output_3x2.txt")) == NULL)
        goto error;

    if ((transformation = load_mat_from_file("3x2/test_mat_3x2.txt")) == NULL)
        goto error;

    char *result = test_transform(input_vector, transformation, expected_output);
    char *test_result = NULL;
    if (result != NULL) {
        char buf[MAX_LEN];
        strcpy(buf, "Error in test transform 3x2: ");
        strcat(buf, result);
        free(result);
        test_result = build_error_message(buf);
    }

    free(input_vector);
    free(expected_output);
    destroy_matrix(transformation);
    return test_result;

    error:
    if (input_vector != NULL) {
        free(input_vector);
    }
    if (expected_output != NULL) {
        free(expected_output);
    }
    if (transformation != NULL) {
        destroy_matrix(transformation);
    }
    return NULL;
}

char *test_transform_2x3(void) {
    double * input_vector = NULL;
    double *expected_output = NULL;
    Matrix *transformation = NULL;
    if ((input_vector = load_vec_from_file("transform_me.txt")) == NULL) 
        goto error;
   
    if ((expected_output = load_vec_from_file("2x3/expected_output_2x3.txt")) == NULL)
        goto error;

    if ((transformation = load_mat_from_file("2x3/test_mat_2x3.txt")) == NULL)
        goto error;

    char *result = test_transform(input_vector, transformation, expected_output);
    char *test_result = NULL;
    if (result != NULL) {
        char buf[MAX_LEN];
        strcpy(buf, "Error in test transform 2x3: ");
        strcat(buf, result);
        free(result);
        test_result = build_error_message(buf);
    }

    free(input_vector);
    free(expected_output);
    destroy_matrix(transformation);
    return test_result;

    error:
    if (input_vector != NULL) {
        free(input_vector);
    }
    if (expected_output != NULL) {
        free(expected_output);
    }
    if (transformation != NULL) {
        destroy_matrix(transformation);
    }
    return NULL;
}

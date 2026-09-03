#include "../../../src/linal/linal.h"
#include <stdio.h>

char * test_matrix_init(void);

char *test_vector_add(void);

int read_next_int(FILE *resource, int *target);

int read_next_double(FILE *resource, double *target);

Matrix *load_mat_from_file(char *file_name);

double *load_vec_from_file(char *file_name);

char *test_transform_2x3(void);

char *test_transform_3x2(void);

char *test_transform_3x3(void);

void test_linal(int *);

char *test_outer_product(void);

char *test_outer_product(void);

char *test_add_matrices(void);

char *test_transpose(void);

char *test_scale_rows_destructive(void);

char *test_scale_rows(void);
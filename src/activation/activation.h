#include "../linal/linal.h"

double relu(double x);
double relu_prime(double x);
double sigmoid(double x);
double sigmoid_prime(double x);
void softmax(double * restrict all_values, int num_values, double * restrict output);
void softmax_destructive(double *all_values, int num_values);

/**
 * @brief computes a matrix representing the partial derivatives of softmax
 * 
 * Computes partial derivative dSi/dvj for row i column j and stores in the output matrix.
 * The output matrix must be a square matrix with both dimensions equal to the size of the input vector.
 * 
 * @param softmax_outputs The pre-computed output of softmax of the original vector.
 * @param output_matrix A matrix that the output will be written to. Must be m x m, where m is the size of softmax_outputs.
*/
void softmax_prime(const double * const restrict softmax_outputs, Matrix * const output_matrix);
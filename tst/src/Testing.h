#include "../../src/linal/linal.h"
#include "..\..\src/multilayer_perceptron/basic/basic.h"
#include <stdio.h>
#define MAX_LEN 10240

#define ASSERT_INT_EQUALS(expected_value, actual_value, test_name) \
   error_message = assert_int_equals(expected_value, actual_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

#define ASSERT_DOUBLE(expected_value, actual_value, error, test_name) \
 error_message = assert_double_approx(expected_value, actual_value, error); \
 if (error_message != NULL) { \
    return enrich_error_message(error_message, test_name); \
 }

#define ASSERT_ACTIVATION_EQUALS(expected_value, actual_value, test_name) \
   error_message = assert_activation_equals(expected_value, actual_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

 #define ASSERT_DOUBLE_GE(expected_value, actual_value, test_name) \
   error_message = assert_double_ge(expected_value, actual_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

#define ASSERT_DOUBLE_LE(expected_value, actual_value, test_name) \
   error_message = assert_double_le(expected_value, actual_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

#define ASSERT_NAN(expected_value, test_name) \
   error_message = assert_nan(expected_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

#define ASSERT_NULL(actual_value, test_name) \
   error_message = assert_null(actual_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

#define ASSERT_NOT_NULL(actual_value, test_name) \
   error_message = assert_not_null(actual_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

#define ASSERT_POINTER_EQUALS(expected_value, actual_value, test_name) \
   error_message = assert_pointer_equals(expected_value, actual_value); \
   if (error_message != NULL) { \
      return enrich_error_message(error_message, test_name); \
   }

 #define RUN_TEST(test_method) \
      error_message = test_method(); \
      if (error_message != NULL) { \
         fprintf(stderr, error_message); \
         free(error_message); \
         error_message = NULL; \
         (*failures)++; \
      } \
      else if (errno) { \
         return; \
      }

extern const char sep[2];

char * assert_int_equals(const int a, const int b);

char * assert_array_equals(const void * const restrict a, const void * const restrict b, const int length, const int size);

char *build_error_message(const char * const error);

char *assert_double_approx(const double expected, const double actual, const double error);

void load_resources_dir(void);

FILE *load_resource_file(const char * const file_name);

int join_path(char * restrict const root, const char * restrict const stem);

int read_line(char * const buf, FILE * const stream, const int maxlen);

char *assert_vector_equals(const Vector * const restrict expected, const Vector * const restrict actual, const double error);

char *enrich_error_message(char * restrict const error_message, const char * restrict const method_name);

char *assert_double_ge(const double expected, const double actual);

char *assert_double_le(const double expected, const double actual);

char *assert_nan(const double actual);

char *assert_null(void * actual);

char *assert_not_null(void *actual);

char *assert_pointer_equals(void *expected, void *actual);

char *assert_activation_equals(Activation expected, Activation actual);
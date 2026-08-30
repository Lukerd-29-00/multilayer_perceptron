#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "../../src/stat/gaussian.h"

void write_double(FILE * const fp, const double value) {
    char buf[16];
    sprintf(buf, "%.4f\n", value);
    size_t len = strlen(buf);
    fwrite(buf, sizeof(char), len, fp);
}

void write_uniform(const size_t count) {
    FILE *fp = fopen("uniform_randoms.txt", "w");
    if (fp == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        const double r = uniform();
        write_double(fp, r);
        if (errno) {
            fclose(fp);
            return;
        }
    }
    fclose(fp);
}

void write_gaussian(const size_t count) {
    FILE *fp = fopen("normal_randoms.txt", "w");
    if (fp == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        const double r = rand_gaussian(0, 1);
        write_double(fp, r);
        if (errno) {
            fclose(fp);
            return;
        }
    }
    fclose(fp);
}

int main(void) {
    write_uniform(1000);
    if (errno) {
        perror("Error writing uniform: ");
    }
    write_gaussian(1000);
    if (errno) {
        perror("Error writing gaussian");
    }
}

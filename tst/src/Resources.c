#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include "Testing.h"

char resources_path[MAX_PATH];
const char resources_dir_name[10] = "resources";
const char sep[2] = "\\";

int read_line(char * const buf, FILE * const stream, const int maxlen) {
    bool file_end = false;
    int read = 0;
    memset(buf, 0, maxlen);
    for (int i = 0; i < maxlen; i++) {
        char c = fgetc(stream);
        if (c == EOF) {
            file_end = true;
            break;
        } 
        buf[i] = c;
        read++;
        if (c == '\n') {
            break;
        }
    }
    if (file_end) {
        int err = ferror(stream);
        if (err) {
            //I/o error
            errno = 447;
            return -2;
        }
        return EOF;
    } else if (buf[read - 1] == '\n') {
        return 0;
    } else {
        //invalid file format
        errno = 448;
        return -2;
    }
}

int join_path(char * restrict const root, const char * restrict const stem) {
    const int len = strlen(root);
    const int stem_len = strlen(stem);

    if (len + stem_len + sizeof(sep) > MAX_PATH) {
        errno = 445;
        return -1;
    }

    strcat(root, sep);
    strcat(root, stem);
    return 0;
}


FILE *load_resource_file(const char * const file_name) {
    char resource_path[MAX_PATH];
    strcpy(resource_path, resources_path);
    if (join_path(resource_path, file_name) == -1) 
        return NULL;

    return fopen(resource_path, "r");

}

void load_resources_dir(void) {
    char exe_path[MAX_PATH];

    if (GetModuleFileName(NULL, exe_path, MAX_PATH) == 0) {
        return;
    }
    char *last_backslash = strrchr(exe_path, '\\');
    if (last_backslash == NULL) {
        errno = 446;
        return;
    }

    *last_backslash = '\0';
    if (join_path(exe_path, resources_dir_name) == -1) {
        return;
    }

    strcpy(resources_path, exe_path);
    
}
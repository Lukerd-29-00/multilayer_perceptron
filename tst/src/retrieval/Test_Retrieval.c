#include "test_retrieval.h"
#include "../../../src/retrieval/retrieval.h"
#include "../Testing.h"
#include <sqlite3.h>
#include <stdio.h>
#include <errno.h>
#include <stddef.h>

#define SETUP_TEST_RETRIEVAL setup(); \
    if (errno != 0) { \
        const char *errmsg =  sqlite3_errstr(errno); \
        printf("%s\n", errmsg); \
        sqlite3_close_v2(test_conn); \
        return NULL; \
    }

const char *learnset = "test_learnset";

const char *create_tables = "CREATE TABLE IF NOT EXISTS TRAINING_RUN (id INTEGER PRIMARY KEY, learnset TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_training_run_learnset ON TRAINING_RUN(learnset); CREATE TABLE IF NOT EXISTS TRAINING_RUN_DATA (input_or_output TEXT NOT NULL, training_run INTEGER NOT NULL, idx INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY(input_or_output, training_run, idx))";

const char *insert_training_run = "INSERT INTO TRAINING_RUN (learnset) VALUES (?1)";
const char *insert_training_run_data = "INSERT INTO TRAINING_RUN_DATA (input_or_output, training_run, idx, value) VALUES (?1, ?2, ?3, ?4)";

sqlite3 *test_conn;

sqlite3_stmt *insert_training_run_stmt;
sqlite3_stmt *insert_training_run_data_stmt;

void setup() {
    int ok = sqlite3_open(":memory:", &test_conn);
    if (ok != SQLITE_OK) {
        return;
    }
   

    ok = sqlite3_exec(
        test_conn,
        create_tables,
        NULL,
        NULL,
        NULL
    );
    if (ok != SQLITE_OK) {
        return;
    }

    ok = sqlite3_prepare_v2(
        test_conn,
        insert_training_run,
        256,
        &insert_training_run_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        return;
    }

    ok = sqlite3_prepare_v2(
        test_conn,
        insert_training_run_data,
        256,
        &insert_training_run_data_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        sqlite3_finalize(insert_training_run_stmt);
        return;
    }

    initialize_retrieval(test_conn);

}

void teardown() {
    sqlite3_finalize(insert_training_run_stmt);
    sqlite3_finalize(insert_training_run_data_stmt);
    sqlite3_close_v2(test_conn);
    stop_retrieval();
}

size_t add_training_run(Training_Run_t training_run, size_t input_values_size, size_t output_values_size) {
    sqlite3_bind_text(insert_training_run_stmt, 1, learnset, 13, SQLITE_STATIC);
    int ok = sqlite3_step(insert_training_run_stmt);
    if (ok != SQLITE_DONE) {
        return FAILURE;
    }

    size_t id = sqlite3_last_insert_rowid(test_conn);

    sqlite3_reset(insert_training_run_stmt);
    sqlite3_clear_bindings(insert_training_run_stmt);

    for (int i = 0; i < input_values_size; i++) {
        sqlite3_bind_text(insert_training_run_data_stmt, 1, "input", 5, SQLITE_STATIC);
        sqlite3_bind_int(insert_training_run_data_stmt, 2, id);
        sqlite3_bind_int(insert_training_run_data_stmt, 3, i);
        sqlite3_bind_double(insert_training_run_data_stmt, 4, training_run.input_values[i]);

        sqlite3_step(insert_training_run_data_stmt);
        sqlite3_reset(insert_training_run_data_stmt);
        sqlite3_clear_bindings(insert_training_run_stmt);

    }

    for (int i = 0; i < output_values_size; i++) {
        sqlite3_bind_text(insert_training_run_data_stmt, 1, "output", 6, SQLITE_STATIC);
        sqlite3_bind_int(insert_training_run_data_stmt, 2, id);
        sqlite3_bind_int(insert_training_run_data_stmt, 3, i);
        sqlite3_bind_double(insert_training_run_data_stmt, 4, training_run.correct_answer[i]);

        sqlite3_step(insert_training_run_data_stmt);
        sqlite3_reset(insert_training_run_data_stmt);
        sqlite3_clear_bindings(insert_training_run_stmt);

    }

    return id;

}

char *test_learnset_size() {
    SETUP_TEST_RETRIEVAL;
    double input_values[1] = {1.5};
    double output_values[1] = {2.5};
    Training_Run_t run = {.input_values = input_values,.correct_answer = output_values};
    int id = add_training_run(run, 1, 1);

    char *error_message = NULL;

    ASSERT_INT_EQUALS(1, learnset_size(learnset), "learnset size");
    teardown();
    return NULL;
}

char *test_load() {
    SETUP_TEST_RETRIEVAL;
    double input_values[1] = {1.5};
    double output_values[1] = {2.5};

    double input_values_2[1] = {1.7};
    double output_values_2[1] = {2.7};
    Training_Run_t run = {.input_values = input_values, .correct_answer = output_values};
    Training_Run_t run_2 = {.input_values = input_values_2, .correct_answer = output_values_2};

    add_training_run(run, 1, 1);
    add_training_run(run_2, 1, 1);

    char *error_message = NULL;

    double output_input_values[1];
    double output_correct_answer[1];
    Training_Run_t loaded_run = {.input_values = output_input_values, .correct_answer = output_correct_answer};

    Training_Run_t runs[1] = {loaded_run};

    load(learnset, 1, 1, runs);
    ASSERT_DOUBLE(1.7, runs[0].input_values[0], 0.01, "test load");
    ASSERT_DOUBLE(2.7, runs[0].correct_answer[0], 0.01, "test load");

    teardown();
    return NULL;
}

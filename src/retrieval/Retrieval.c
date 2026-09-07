#include "retrieval.h"
#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

const char *sample = "select id FROM TRAINING_RUN WHERE learnset = ?1 LIMIT ?2 OFFSET ?3";
const char *learnset_size_q = "SELECT COUNT(*) FROM TRAINING_RUN WHERE learnset = ?1";
const char *select_input_data = "select value FROM TRAINING_RUN_DATA WHERE input_or_output = 'input' AND training_run = ?1 ORDER BY idx";
const char *select_output_data = "select value FROM TRAINING_RUN_DATA WHERE input_or_output = 'output' AND training_run = ?1 ORDER BY idx";
sqlite3 *conn;

sqlite3_stmt *learnset_size_stmt;
sqlite3_stmt *sample_stmt;
sqlite3_stmt *select_input_data_stmt;
sqlite3_stmt *select_output_data_stmt;

void initialize_retrieval(sqlite3 *input_conn) {
    conn = input_conn;
    int ok = sqlite3_prepare_v2(
        conn,
        learnset_size_q,
        128,
        &learnset_size_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(conn);
        return;
    }

    ok = sqlite3_prepare_v2(
        conn,
        sample,
        128,
        &sample_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(conn);
        sqlite3_finalize(learnset_size_stmt);
        return;
    }

    ok = sqlite3_prepare_v2(
        conn,
        select_input_data,
        128,
        &select_input_data_stmt,
        NULL
    );
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(conn);
        sqlite3_finalize(learnset_size_stmt);
        sqlite3_finalize(sample_stmt);
        return;
    }

    ok = sqlite3_prepare_v2(
        conn,
        select_output_data,
        128,
        &select_output_data_stmt,
        NULL
    );

    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(conn);
        sqlite3_finalize(learnset_size_stmt);
        sqlite3_finalize(sample_stmt);
        sqlite3_finalize(select_input_data_stmt);
    }

}

void stop_retrieval() {
    sqlite3_finalize(learnset_size_stmt);
    sqlite3_finalize(sample_stmt);
    sqlite3_finalize(select_input_data_stmt);
    sqlite3_finalize(select_output_data_stmt);
}

size_t learnset_size(const char *learnset) {
    int ok = sqlite3_bind_text(learnset_size_stmt, 1, learnset, strlen(learnset), SQLITE_STATIC);
    if (ok != SQLITE_OK) {
        return FAILURE;
    }

    ok = sqlite3_step(learnset_size_stmt);
    if (ok == SQLITE_DONE) {
        errno = NO_ROWS_FOUND;
        return FAILURE;
    } else if (ok != SQLITE_ROW) {
        errno = sqlite3_errcode(conn);
        return FAILURE;
    }
    
    size_t output = sqlite3_column_int(learnset_size_stmt, 0);

    ok = sqlite3_step(learnset_size_stmt);
    if (ok == SQLITE_ROW) {
        errno = EXCESS_ROWS_FOUND;
        return FAILURE;
    } else if (ok != SQLITE_DONE) {
        errno = sqlite3_errcode(conn);
        return FAILURE;
    }

    sqlite3_reset(learnset_size_stmt);
    sqlite3_clear_bindings(learnset_size_stmt);
    return output;
}

void load(const char *learnset, const size_t sample_idx, const size_t sample_size, Training_Run_t *training_runs) {
    sqlite3_bind_text(sample_stmt, 1, learnset, strlen(learnset), SQLITE_STATIC);
    sqlite3_bind_int(sample_stmt, 2, sample_size);
    sqlite3_bind_int(sample_stmt, 3, sample_idx);

    int i = 0;
    int is_done = sqlite3_step(sample_stmt);
    while (is_done == SQLITE_ROW) {
        int id = sqlite3_column_int(sample_stmt, 0);
        load_training_run_data(id, &training_runs[i]);
        if (sqlite3_errcode(conn) != 0) {
            break;
        }
        is_done = sqlite3_step(sample_stmt);
        i++;
    }

    sqlite3_reset(sample_stmt);
    sqlite3_clear_bindings(sample_stmt);
    if (sqlite3_errcode(conn) != 0) {
        errno = sqlite3_errcode(conn);
    }
}




void load_training_run_data(int id, Training_Run_t *training_run) {
    sqlite3_bind_int(select_input_data_stmt, 1, id);
    int i = 0;
    int is_done = sqlite3_step(select_input_data_stmt);
    while (is_done == SQLITE_ROW) {
        training_run->input_values[i] = sqlite3_column_double(select_input_data_stmt, 0);
        is_done = sqlite3_step(select_input_data_stmt);
        i++;
    }

    int ok = sqlite3_reset(select_input_data_stmt);
    sqlite3_clear_bindings(select_input_data_stmt);
    if (ok != SQLITE_OK) {
        errno = sqlite3_errcode(conn);
        return;
    }

    
    sqlite3_bind_int(select_output_data_stmt, 1, id);
    i = 0;
    is_done = sqlite3_step(select_output_data_stmt);
    while (is_done != SQLITE_DONE) {
        training_run->correct_answer[i] = sqlite3_column_double(select_output_data_stmt, 0);
        is_done = sqlite3_step(select_output_data_stmt);
        i++;
    }

    sqlite3_reset(select_output_data_stmt);
    sqlite3_clear_bindings(select_output_data_stmt);
}


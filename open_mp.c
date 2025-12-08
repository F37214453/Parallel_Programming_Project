#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <dirent.h>

#define MAX_LINE_LEN 256
#define MIN_PRICE 0.01
#define MAX_PRICE 10000.0
#define MIN_YEAR_GLOBAL 1900
#define MAX_YEAR_GLOBAL 2100
#define MAX_DECADES (((MAX_YEAR_GLOBAL - MIN_YEAR_GLOBAL) / 10) + 1)

typedef struct {
    char date[20];
    double open, high, low, close, volume;
} StockData;

int read_csv(const char *filename, StockData **data_out) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;

    char line[MAX_LINE_LEN];
    int count = 0, capacity = 0;
    StockData *data = NULL;

    fgets(line, sizeof(line), file); // skip header

    while (fgets(line, sizeof(line), file)) {
        if (count >= capacity) {
            int new_cap = (capacity == 0) ? 1024 : capacity * 2;
            StockData *tmp = realloc(data, new_cap * sizeof(StockData));
            if (!tmp) { free(data); fclose(file); *data_out = NULL; return 0; }
            data = tmp;
            capacity = new_cap;
        }
        double adj_temp;
        if (sscanf(line, "%19[^,],%lf,%lf,%lf,%lf,%lf,%lf",
            data[count].date, &data[count].open, &data[count].high,
            &data[count].low, &data[count].close, &adj_temp, &data[count].volume) == 7) {
            count++;
        }
    }
    fclose(file);
    *data_out = data;
    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <stocks_directory>\n", argv[0]);
        return 1;
    }

    const char *dirpath = argv[1];
    DIR *dir = opendir(dirpath);
    if (!dir) { fprintf(stderr, "Err: %s\n", dirpath); return 1; }

    struct dirent *entry;
    char **file_list = NULL;
    int file_count = 0, file_cap = 0;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        size_t len = strlen(entry->d_name);
        if (len < 4 || strcmp(entry->d_name + len - 4, ".csv") != 0) continue;

        if (file_count >= file_cap) {
            file_cap = (file_cap == 0) ? 128 : file_cap * 2;
            file_list = realloc(file_list, file_cap * sizeof(char*));
        }
        char fullpath[1024];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
        file_list[file_count++] = strdup(fullpath);
    }
    closedir(dir);

    double global_sum_avg[MAX_DECADES] = {0};
    long   global_count_rows[MAX_DECADES] = {0};
    double global_sum_ret[MAX_DECADES] = {0};
    double global_sum_ret_sq[MAX_DECADES] = {0};
    long   global_count_ret[MAX_DECADES] = {0};

    double total_calc_time = 0.0;
    long total_rows = 0;

    printf("\nMarket Summary by Decade:\n");

    #pragma omp parallel
    {
        double loc_sum_avg[MAX_DECADES] = {0};
        long   loc_rows[MAX_DECADES] = {0};
        double loc_sum_ret[MAX_DECADES] = {0};
        double loc_sum_ret_sq[MAX_DECADES] = {0};
        long   loc_ret[MAX_DECADES] = {0};

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < file_count; i++) {
            StockData *data = NULL;
            int n = read_csv(file_list[i], &data);
            if (n <= 1 || !data) { free(file_list[i]); continue; }

            double t1 = omp_get_wtime();

            for (int j = 0; j < n; j++) {
                int year;
                if (sscanf(data[j].date, "%d", &year) != 1) continue;
                if (year < MIN_YEAR_GLOBAL || year > MAX_YEAR_GLOBAL) continue;
                int d = (year - MIN_YEAR_GLOBAL) / 10;

                double o = data[j].open, c = data[j].close;
                if (o >= MIN_PRICE && o <= MAX_PRICE && c >= MIN_PRICE && c <= MAX_PRICE) {
                    loc_sum_avg[d] += (data[j].open + data[j].high + data[j].low + data[j].close)/4.0;
                    loc_rows[d]++;
                }
            }

            for (int j = 0; j < n - 1; j++) {
                int year;
                if (sscanf(data[j].date, "%d", &year) != 1) continue;
                if (year < MIN_YEAR_GLOBAL || year > MAX_YEAR_GLOBAL) continue;
                int d = (year - MIN_YEAR_GLOBAL) / 10;

                double p = data[j].close;
                double q = data[j+1].close;
                if (p >= MIN_PRICE && q >= MIN_PRICE && p != 0) {
                    double r = (q - p)/p;
                    if (fabs(r) <= 1.0) {
                        loc_sum_ret[d] += r;
                        loc_sum_ret_sq[d] += r*r;
                        loc_ret[d]++;
                    }
                }
            }

            double t2 = omp_get_wtime();
            #pragma omp atomic
            total_calc_time += (t2 - t1);
            #pragma omp atomic
            total_rows += n;

            free(data);
            free(file_list[i]);
        }

        #pragma omp critical
        {
            for (int d = 0; d < MAX_DECADES; d++) {
                global_sum_avg[d]    += loc_sum_avg[d];
                global_count_rows[d] += loc_rows[d];
                global_sum_ret[d]    += loc_sum_ret[d];
                global_sum_ret_sq[d] += loc_sum_ret_sq[d];
                global_count_ret[d]  += loc_ret[d];
            }
        }
    }

    free(file_list);

    for (int d = 0; d < MAX_DECADES; d++) {
        long rows = global_count_rows[d];
        if (rows == 0) continue;

        double mean_price = global_sum_avg[d]/rows;
        long rets = global_count_ret[d];
        double vol = 0, mean_r = 0, annual_r = 0;

        if (rets > 0) {
            mean_r = global_sum_ret[d]/rets;
            double mean_r2 = global_sum_ret_sq[d]/rets;
            double var = mean_r2 - mean_r*mean_r;
            if (var < 0) var = 0;
            vol = sqrt(var);
            annual_r = pow(1+mean_r, 252.0)-1;
        }

        int start_year = MIN_YEAR_GLOBAL + d*10;
        int end_year = start_year + 9;

        printf("Decade %d-%d:\n", start_year, end_year);
        printf("Rows used:\n%ld\n", rows);
        printf("Mean market price:\n%.4f\n", mean_price);
        printf("Market volatility:\n%.4f (%.4f%%)\n", vol, vol*100);
        if (rets > 0) {
            printf("Mean daily return:\n%.6f (%.4f%%)\n", mean_r, mean_r*100);
            printf("Approx annual return:\n%.6f (%.4f%%)\n", annual_r, annual_r*100);
        } else {
            printf("Mean daily return:\nN/A\nApprox annual return:\nN/A\n");
        }
        printf("\n");
    }

    printf("Execution time (parallel sum of threads): %.6f seconds\n", total_calc_time);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <dirent.h>

#define MAX_LINE_LEN 256

// logical price for cleaning the data
#define MIN_PRICE 0.01
#define MAX_PRICE 10000.0

// Allowed range for the years to be wide and safe
#define MIN_YEAR_GLOBAL 1900
#define MAX_YEAR_GLOBAL 2100
#define MAX_DECADES (((MAX_YEAR_GLOBAL - MIN_YEAR_GLOBAL) / 10) + 1)

// data structure representing one row of stock data
typedef struct {
    char date[20];
    double open, high, low, close, volume;
} StockData;

// compute daily avg price from OHLC
double daily_average(const StockData *s) {
    return (s->open + s->high + s->low + s->close) / 4.0;
}

// compute daily returns (curr - prev) / prev
// Protect to not divied by zero 
double daily_return(double prev, double curr) {
    if (prev == 0.0) return 0.0;
    return (curr - prev) / prev;
}

// Read the data form the CSV files to insert it to the dynamic array of StocksData
// Returns number of rows; set *data_out
// expected CSV format
// data we read is open, high, low, close, adj_close, volume
int read_csv(const char *filename, StockData **data_out) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        *data_out = NULL;
        return 0;
    }

    char line[MAX_LINE_LEN];
    int count = 0, capacity = 0;
    StockData *data = NULL;

    // skip header
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        // scaling if needed(Grow array )
        if (count >= capacity) {
            int new_cap = (capacity == 0) ? 1024 : capacity * 2;
            StockData *tmp = realloc(data, new_cap * sizeof(StockData));
            if (!tmp) {
                fprintf(stderr, "Memory allocation failed\n");
                free(data);
                fclose(file);
                *data_out = NULL;
                return 0;
            }
            data = tmp;
            capacity = new_cap;
        }

        double adj_temp;
        
        // read all 7 fields
        int parsed = sscanf(
            line,
            "%19[^,],%lf,%lf,%lf,%lf,%lf,%lf",
            data[count].date,
            &data[count].open,
            &data[count].high,
            &data[count].low,
            &data[count].close,
            &adj_temp,
            &data[count].volume
        );

        if (parsed == 7) {
            count++;
        }
    }

    fclose(file);
    *data_out = data;
    return count;
}

// cleans data, groups statistics by decade, price results 
int main(int argc, char *argv[]) {

    double total_time_exe = 0.0;
    double start;
    double end;
    if (argc != 2) {
        printf("Usage: %s <stocks_directory>\n", argv[0]);
        return 1;
    }

    const char *dirpath = argv[1];
    DIR *dir = opendir(dirpath);
    if (!dir) {
        fprintf(stderr, "Cannot open directory: %s\n", dirpath);
        return 1;
    }

    printf("\nSerial Stock Analysis - Market Metrics by Decade (Cleaned)\n");
    printf("Directory: %s\n", dirpath);
    printf("============================================================\n\n");

    struct dirent *entry;
    char filepath[1024];

  // Accumulators per decade 
    double sum_avg_decade[MAX_DECADES]    = {0.0};
    long   count_rows_decade[MAX_DECADES] = {0};

    double sum_ret_decade[MAX_DECADES]    = {0.0};
    double sum_ret_sq_decade[MAX_DECADES] = {0.0};
    long   count_ret_decade[MAX_DECADES]  = {0};

    int global_min_year = 9999;
    int global_max_year = 0;


    // Iterate through files
    while ((entry = readdir(dir)) != NULL) {

        const char *name = entry->d_name;

        // ingro the file that has '..'
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
            continue;

        // take only the file with  .csv
        size_t len = strlen(name);
        if (len < 4 || strcmp(name + len - 4, ".csv") != 0)
            continue;

        snprintf(filepath, sizeof(filepath), "%s/%s", dirpath, name);

        StockData *data = NULL;
        int n = read_csv(filepath, &data);
        if (n <= 1 || !data) {
            if (data) free(data);
            continue;
        }

        start = omp_get_wtime();

        // track min & max year in data
        for (int i = 0; i < n; i++) {
            int year = 0;
            sscanf(data[i].date, "%d", &year);
            if (year < global_min_year) global_min_year = year;
            if (year > global_max_year) global_max_year = year;
        }

        // collective daily average prices per decade
        for (int i = 0; i < n; i++) {
            double o = data[i].open;
            double h = data[i].high;
            double l = data[i].low;
            double c = data[i].close;

            int year = 0;
            sscanf(data[i].date, "%d", &year);

            if (year < MIN_YEAR_GLOBAL || year > MAX_YEAR_GLOBAL)
                continue;

            int decade_index = (year - MIN_YEAR_GLOBAL) / 10;
            if (decade_index < 0 || decade_index >= MAX_DECADES)
                continue;

            //Clean unrealistic prices
            if (o >= MIN_PRICE && o <= MAX_PRICE &&
                h >= MIN_PRICE && h <= MAX_PRICE &&
                l >= MIN_PRICE && l <= MAX_PRICE &&
                c >= MIN_PRICE && c <= MAX_PRICE)
            {
                double avg = (o + h + l + c) / 4.0;
                sum_avg_decade[decade_index]    += avg;
                count_rows_decade[decade_index] += 1;
            }
        }

        // collective daily returns prices per decade
        for (int i = 0; i < n - 1; i++) {
            double p = data[i].close;
            double q = data[i + 1].close;

            int year = 0;
            sscanf(data[i].date, "%d", &year);

            if (year < MIN_YEAR_GLOBAL || year > MAX_YEAR_GLOBAL)
                continue;

            int decade_index = (year - MIN_YEAR_GLOBAL) / 10;
            if (decade_index < 0 || decade_index >= MAX_DECADES)
                continue;

            if (p >= MIN_PRICE && p <= MAX_PRICE &&
                q >= MIN_PRICE && q <= MAX_PRICE &&
                p != 0.0)
            {
                double r = (q - p) / p;

                // exclude extreme outliers (> 100 daily move)
                if (fabs(r) > 1.0)
                    continue;

                sum_ret_decade[decade_index]    += r;
                sum_ret_sq_decade[decade_index] += r * r;
                count_ret_decade[decade_index]  += 1;
            }
        }
        end = omp_get_wtime();

        // total computiation time 
        total_time_exe += (end - start);
        free(data);
    }

    closedir(dir);


    // print the analysis results by decade
    printf("Market Summary by Decade:\n");
    printf("------------------------------------------------------------\n");

    int first_decade = (global_min_year / 10) * 10;
    int last_decade  = 2010;  // آخر فترة: 2010–2020

    for (int decade_start = first_decade; decade_start <= last_decade; decade_start += 10) {
        int idx = (decade_start - MIN_YEAR_GLOBAL) / 10;
        if (idx < 0 || idx >= MAX_DECADES)
            continue;

        long rows = count_rows_decade[idx];
        long rets = count_ret_decade[idx];

        if (rows == 0 && rets == 0)
            continue; 

        double mean_price = 0.0;
        double vol = 0.0;
        double mean_r = 0.0;
        double annual_r = 0.0;

        if (rows > 0) {
            mean_price = sum_avg_decade[idx] / (double)rows;
        }

        if (rets > 0) {
            mean_r  = sum_ret_decade[idx]    / (double)rets;  // daily average
            double mean_r2 = sum_ret_sq_decade[idx] / (double)rets;
            double var = mean_r2 - mean_r * mean_r;
            if (var < 0.0) var = 0.0;
            vol = sqrt(var);

                  annual_r = mean_r * 252.0; // approximate annualization

        }

        int decade_end = (decade_start == 2010) ? 2020 : (decade_start + 9);

        printf("Decade %d-%d:\n", decade_start, decade_end);
        printf("  Rows used:             %ld\n", rows);
        printf("  Mean market price:     %.4f\n", mean_price);
        printf("  Market volatility:     %.4f (%.4f%%)\n",
               vol, vol * 100.0);

        if (rets > 0) {
            printf("  Mean daily return:     %.6f (%.4f%%)\n",
                   mean_r, mean_r * 100.0);
            printf("  Approx annual return:  %.6f (%.4f%%)\n\n",
                   annual_r, annual_r * 100.0);
        } else {
            printf("  Mean daily return:     N/A\n");
            printf("  Approx annual return:  N/A\n\n");
        }
    }

    printf("Execution time (serial): %.6f seconds\n", total_time_exe);

    return 0;
}

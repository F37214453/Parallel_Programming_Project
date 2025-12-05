#include <stdio.h>
#include <stdlib.h> // for malloc, free
#include <string.h>
#include <math.h>
#include <omp.h>   

#define MAX_LINE_LEN 256  // max line length
#define MAX_DAYS     1000 // default max number of days to read

typedef struct {
    char date[20]; 
    double open, high, low, close, volume; // Price and volume columns
} StockData;

// daily_average : summary for daily price (simple helper, stays serial) important
double daily_average(StockData s) {
    return (s.open + s.high + s.low + s.close) / 4.0;
}

// daily_return: explains the change between the previous day's close
double daily_return(double prev_close, double curr_close) {
    if (prev_close == 0) return 0.0;
    return (curr_close - prev_close) / prev_close;
}

int read_csv(const char *filename, StockData *data, int max_days) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;

    char line[MAX_LINE_LEN];
    int count = 0;
    fgets(line, sizeof(line), file); // Skip Header

    while (fgets(line, sizeof(line), file) && count < max_days) {
        if (sscanf(line, "%19[^,],%lf,%lf,%lf,%lf,%lf",
                   data[count].date, &data[count].open, &data[count].high,
                   &data[count].low, &data[count].close, &data[count].volume) == 6) {
            count++;
        }
    }
    fclose(file);
    return count;
}

int main(int argc, char *argv[]) {

    int max_days = MAX_DAYS;
    int file_start_index = 1;

    if (argc >= 2) {
        char *endptr;
        long tmp = strtol(argv[1], &endptr, 10);
        if (*endptr == '\0' && tmp > 0) {
            max_days = (int)tmp;
            file_start_index = 2;
        }
    }

    printf("\nAnalysis result of stock files (OpenMP version)\n");
    printf("===========================================\n\n");

    for (int f = file_start_index; f < argc; f++) {
        const char *filename = argv[f];
        StockData *data = malloc(sizeof(StockData) * max_days);
        
       
        int n = read_csv(filename, data, max_days);
        if (n <= 1) {
            printf("Not enough data in file: %s\n", filename);
            free(data);
            continue;
        }

        double start_time = omp_get_wtime();

        double avg_sum = 0.0;
        double sum_ret = 0.0;
        double sum_ret_sq = 0.0;

        // We use one parallel region to avoid starting/stopping threads multiple times.
        #pragma omp parallel default(none) \
                shared(n, data, avg_sum, sum_ret, sum_ret_sq)
        {
            #pragma omp for reduction(+:avg_sum)
            for (int i = 0; i < n; i++) {
                avg_sum += daily_average(data[i]);
            }

            #pragma omp for reduction(+:sum_ret, sum_ret_sq)
            for (int i = 0; i < n - 1; i++) {
                double r = daily_return(data[i].close, data[i+1].close);
                sum_ret += r;
                sum_ret_sq += r * r;
            }
        } 
 
        // calculate the standard deviation of the daily returns (volatility).
        double avg_price = avg_sum / n;
        double mean_ret = sum_ret / (n - 1);
        double mean_sq = sum_ret_sq / (n - 1);
        double variance = mean_sq - (mean_ret * mean_ret);
        if (variance < 0) variance = 0;
        double volatility = sqrt(variance);


        double end_time = omp_get_wtime();
        double elapsed = end_time - start_time;

        printf("Symbol: %s\n", filename);
        printf("Records loaded: %d\n", n);
        printf("Average daily price (USD): %.4f\n", avg_price);
        printf("Volatility (std. dev of returns): %.6f\n", volatility);
        printf("Volatility (percentage): %.4f%%\n", volatility * 100);
        printf("Execution time (computation only): %.6f seconds\n", elapsed);
        printf("---------------------------------------------\n");

        free(data);
    }

    return 0;
}
// gcc -O3 -fopenmp open_mp.c -o omp
// export OMP_NUM_THREADS=2
// export OMP_NUM_THREADS=4
// export OMP_NUM_THREADS=8
// export OMP_NUM_THREADS=16
// export OMP_SCHEDULE="static,1000"
// export OMP_SCHEDULE="dynamic,1000"
// export OMP_SCHEDULE="guided,1000"   
// ./omp stock_data.csv

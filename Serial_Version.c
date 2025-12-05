/***************************************************************
 * File: stock_analysis_serial.c
 * Project: Stock Market Analysis (Serial Version)
 *
 * Description:
 * This program analyzes stock price data by computing:
 *    - Daily average price
 *    - Daily returns (percentage change)
 *    - Price volatility (standard deviation of returns)
 *
 * The serial version is a baseline.
 *
 * Input:
 *    One or more CSV files, each containing:
 *      Date, Open, High, Low, Close, Volume
 *    Optionally, a maximum number of days [max_days] can be provided.
 *
 * Output:
 *    - Number of records loaded
 *    - Average daily price
 *    - Volatility (std. deviation)
 *    - Execution time (computation only)
 ****************************************************************/

#include <stdio.h>
#include <stdlib.h> // for malloc, free
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX_LINE_LEN 256  // max line length
#define MAX_DAYS     1000 // default max number of days to read

typedef struct {
    char date[20]; // we put 20 because most date formats fit within this length
    double open, high, low, close, volume; // Price and volume columns
} StockData;

// daily_average : summary for daily price
double daily_average(StockData s) {
    return (s.open + s.high + s.low + s.close) / 4.0;
}

// daily_return: explains the change between the previous day's close
// and the current day's close (important for stock movement analysis)
double daily_return(double prev_close, double curr_close) {
    if (prev_close == 0.0) return 0.0;
    return (curr_close - prev_close) / prev_close;
}

// to read the file
int read_csv(const char *filename, StockData *data, int max_days) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return 0;
    }

    char line[MAX_LINE_LEN];
    int count = 0;

    // skips the title
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file) && count < max_days) {
        // Format: Date,Open,High,Low,Close,Volume
        if (sscanf(line, "%19[^,],%lf,%lf,%lf,%lf,%lf",
                  data[count].date,
                  &data[count].open,
                  &data[count].high,
                  &data[count].low,
                  &data[count].close,
                  &data[count].volume) == 6)
        {
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

    if (file_start_index >= argc) {
        printf("Usage: %s [max_days] <file1.csv> <file2.csv> ... <fileN.csv>\n", argv[0]);
        return 1;
    }

    printf("\nAnalysis result of stock filess\n", argc - file_start_index);
    printf("===========================================\n\n");

    for (int f = file_start_index; f < argc; f++) {
        const char *filename = argv[f];

        StockData *data = malloc(sizeof(StockData) * max_days);
        if (!data) {
            perror("Memory allocation failed");
            continue;
        }

        int n = read_csv(filename, data, max_days);
        if (n <= 1) {
            printf("File: %s Not enough data in file  rows=%d\n", filename, n);
            free(data);
            continue;
        }

        double start_time = omp_get_wtime();

        double avg_sum = 0.0; // to calculate the cumulative daily average
        for (int i = 0; i < n; i++) {
            avg_sum += daily_average(data[i]);
        }
        
        int num_returns = n - 1;
        double total_sum_ret = 0.0;
        double total_sum_ret_sq = 0.0;

        for (int i = 0; i < num_returns; i++){
            double r = daily_return(data[i].close, data[i+1].close);
            total_sum_ret += r;
            total_sum_ret_sq += r * r; 
        }

        double end_time = omp_get_wtime();
        double elapsed = end_time - start_time;

        double avg_price = avg_sum / (double)n;

        double mean_ret = total_sum_ret / (double)num_returns;
        double mean_sq  = total_sum_ret_sq / (double)num_returns;
        
        // Variance = E[x^2] - (E[x])^2
        double variance = mean_sq - (mean_ret * mean_ret);
        if (variance < 0) variance = 0;
        double volatility = sqrt(variance);

        
        printf("File: %s\n", filename);
        printf("Days loaded: %d\n", n);
        printf("Average price: %.4f\n", avg_price);
        printf("Volatility: %.6f\n", volatility);
        printf("Volatility (percentage): %.4f%%\n", volatility * 100);
        printf("Serial time:   %.6f seconds\n", elapsed);
        printf("-------------------------------------------\n");

        free(data);
    }

    return 0;
}

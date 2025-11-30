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
    if (prev_close == 0) return 0.0;
    return (curr_close - prev_close) / prev_close;
}

// compute_volatility : calculate the standard deviation of the daily returns (volatility).
double compute_volatility(double *returns, int n) {
    if (n <= 1) return 0.0;
    double sum = 0.0, mean, variance = 0.0;
    for (int i = 0; i < n; i++) sum += returns[i];
    mean = sum / n;
    for (int i = 0; i < n; i++) variance += pow(returns[i] - mean, 2);
    return sqrt(variance / n);
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
        sscanf(line, "%[^,],%lf,%lf,%lf,%lf,%lf",
               data[count].date, &data[count].open, &data[count].high,
               &data[count].low, &data[count].close, &data[count].volume);
        count++;
    }

    fclose(file);
    return count;
}

int main(int argc, char *argv[]) {
    

    if (argc < 2) {
        printf("Usage: %s [max_days] <file1.csv> <file2.csv> ... <fileN.csv>\n", argv[0]);
        return 1;
    }

    int max_days = MAX_DAYS;
    int file_start_index = 1;

    
    if (argc >= 3) {
        char *endptr;
        long tmp = strtol(argv[1], &endptr, 10);
        if (*endptr == '\0' && tmp > 0) {
            max_days = (int)tmp;
            file_start_index = 2;
        }
    }

    if (argc <= file_start_index) {
        printf("Usage: %s [max_days] <file1.csv> <file2.csv> ... <fileN.csv>\n", argv[0]);
        return 1;
    }

    printf("\nAnalysis result of stock filess\n", argc - file_start_index);
    printf("===========================================\n\n");

    for (int f = file_start_index; f < argc; f++) {
        const char *filename = argv[f];

        StockData *data = malloc(sizeof(StockData) * max_days);
        double *returns = malloc(sizeof(double) * max_days);
        if (!data || !returns) {
            printf("Memory allocation failed for file: %s\n", filename);
            free(data);
            free(returns);
            continue;
        }

        int n = read_csv(filename, data, max_days);
        if (n <= 1) {
            printf("Not enough data in file: %s\n", filename);
            free(data);
            free(returns);
            continue;
        }

        // symbol name
        char symbol[32];
        strcpy(symbol, filename);
        char *dot = strchr(symbol, '.');
        if (dot) *dot = '\0';

        double start_time = omp_get_wtime();

        double avg_sum = 0.0; // to calculate the cumulative daily average
        for (int i = 0; i < n; i++) {
            avg_sum += daily_average(data[i]);
            if (i > 0)
                returns[i - 1] = daily_return(data[i - 1].close, data[i].close);
        }

        double overall_avg = avg_sum / n;
        double volatility = compute_volatility(returns, n - 1);

        double end_time = omp_get_wtime();
        double elapsed = end_time - start_time;

        printf("Symbol: %s\n", symbol);
        printf("Records loaded: %d\n", n);
        printf("Average daily price (USD): %.4f\n", overall_avg);
        printf("Volatility (std. dev of returns): %.6f\n", volatility);
        printf("Volatility (percentage): %.4f%%\n", volatility * 100);
        printf("Execution time (computation only): %.6f seconds\n", elapsed);
        printf("---------------------------------------------\n");

        free(data);
        free(returns);
    }

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_DAYS 1000000

typedef struct {
    double open, high, low, close;
} StockData;

// calculate average
double daily_average(StockData s) {
    return (s.open + s.high + s.low + s.close) / 4.0;
}

// calculate daily return
double daily_return(double prev_close, double curr_close) {
    if (prev_close == 0) return 0.0;
    return (curr_close - prev_close) / prev_close;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    int n; // total records count
    StockData *data = NULL;

    if (rank == 0) {
        // read data from CSV (simulate loaded data)
        FILE *file = fopen("stock_data.csv", "r");
        if (!file) {
            printf("Error: cannot open file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // skip header
        char line[256];
        fgets(line, sizeof(line), file);

        data = malloc(sizeof(StockData) * MAX_DAYS);
        n = 0;

        while (fgets(line, sizeof(line), file) && n < MAX_DAYS) {
            sscanf(line, "%*[^,],%lf,%lf,%lf,%lf,%*lf",
                   &data[n].open, &data[n].high, &data[n].low, &data[n].close);
            n++;
        }
        fclose(file);
        printf("Total records read: %d\n", n);
    }

    // broadcast number of records to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // determine local chunk size
    int chunk = n / size;
    StockData *local_data = malloc(sizeof(StockData) * chunk);

    // distribute equal-sized blocks of data
    MPI_Scatter(data, chunk * sizeof(StockData), MPI_BYTE,
                local_data, chunk * sizeof(StockData), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // synchronize and start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // local computation
    double local_sum_avg = 0.0;
    double *local_returns = malloc(sizeof(double) * (chunk - 1));

    for (int i = 0; i < chunk; i++) {
        local_sum_avg += daily_average(local_data[i]);
        if (i > 0)
            local_returns[i - 1] = daily_return(local_data[i - 1].close, local_data[i].close);
    }

    // compute local volatility
    double sum_ret = 0.0;
    for (int i = 0; i < chunk - 1; i++) sum_ret += local_returns[i];
    double mean_ret = sum_ret / (chunk - 1);

    double var = 0.0;
    for (int i = 0; i < chunk - 1; i++)
        var += pow(local_returns[i] - mean_ret, 2);
    double local_vol = sqrt(var / (chunk - 1));

    // reduction to compute global averages and volatility
    double global_sum_avg = 0.0;
    MPI_Reduce(&local_sum_avg, &global_sum_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_vol = 0.0;
    MPI_Reduce(&local_vol, &global_vol, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        double avg_price = global_sum_avg / n;
        double avg_vol = global_vol / size;

        printf("\n===== MPI Stock Analysis Results =====\n");
        printf("Total Processes: %d\n", size);
        printf("Average Daily Price: %.4f\n", avg_price);
        printf("Average Volatility: %.6f\n", avg_vol);
        printf("Execution Time: %.6f seconds\n", end_time - start_time);
        printf("=====================================\n");
    }

    free(local_data);
    free(local_returns);
    if (rank == 0) free(data);

    MPI_Finalize();
    return 0;
}


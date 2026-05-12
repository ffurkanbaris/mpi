#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void read_matrix(const char* filename, double** matrix, int* N) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Hata: %s dosyasi acilamadi!\n", filename);
        exit(1);
    }
    fscanf(file, "%d", N);
    *matrix = (double*)malloc((*N) * (*N) * sizeof(double));
    for (int i = 0; i < (*N) * (*N); i++) {
        fscanf(file, "%lf", &(*matrix)[i]);
    }
    fclose(file);
}

int main(int argc, char** argv) {
    int rank, size, N;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = NULL, *local_C = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        read_matrix("a-v2.txt", &A, &N);
        read_matrix("b-v2.txt", &B, &N);
        C = (double*)malloc(N * N * sizeof(double));
    }

    // Boyutu (N) tüm process'lere gönder
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B = (double*)malloc(N * N * sizeof(double));
    }

    // B matrisinin tamamı tüm process'lere lazım, bu yüzden Broadcast ediyoruz
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Her process'e düşecek satır sayısı
    int rows_per_process = N / size;
    local_A = (double*)malloc(rows_per_process * N * sizeof(double));
    local_C = (double*)calloc(rows_per_process * N, sizeof(double));

    // Zaman ölçümünü başlat 
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start_time = MPI_Wtime();

    // A matrisinin satırlarını process'lere dağıt (Scatter)
    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, local_A, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Çarpım işlemi: local_C = local_A x B
    for (int i = 0; i < rows_per_process; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Sonuçları Master process'te C matrisinde topla (Gather)
    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE, C, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("C MPI Süresi: %f saniye\n", end_time - start_time);
        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}

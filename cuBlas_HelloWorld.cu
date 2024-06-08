#include <stdio.h> // Inclui a biblioteca padrão de entrada e saída
#include <cublas_v2.h> // Inclui a biblioteca cuBLAS para operações de álgebra linear
#include <curand_kernel.h> // Inclui a biblioteca CURAND para geração de números aleatórios


#define TAMANHO_MATRIZ 8 // Multiplos de 8 para usar Tensor Cores em FP64

// Função para inicializar uma matriz identidade
__global__ void init_identity(double *a, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        a[idx*n + idx] = 1.0;
    }
}

// Função para inicializar uma matriz de uns
__global__ void init_unit(double *b, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            b[idx*n + j] = 1.0;
        }
    }
}

// Função para inicializar uma matriz com números aleatórios
__global__ void init_random(double *a, int n) {
    int idx = threadIdx.x;
    curandState state;
    curand_init(1234, idx, 0, &state);

    if (idx < n) {
        for (int j = 0; j < n; j++) {
            a[idx*n + j] = curand_uniform_double(&state);
        }
    }
}

int main(int argc, char** argv) { // Função principal
    cublasHandle_t handle; // Declara um handle para a biblioteca cuBLAS

    const int N = TAMANHO_MATRIZ; // Define o tamanho da matriz (N x N)
    double *d_A, *d_B, *d_C, *d_D, *d_E; // Declara ponteiros para as matrizes A, B, C, D, E na memória do dispositivo
    
    // C = alpha * (A * B) + beta * (A * B)
    const double alpha = 1.0; // Define o valor de alpha para a operação de multiplicação de matrizes
    const double beta = 0.0; // Define o valor de beta para a operação de multiplicação de matrizes

    cudaMalloc((void**)&d_A, N * N * sizeof(d_A[0])); // Aloca memória na GPU para a matriz A
    cudaMalloc((void**)&d_B, N * N * sizeof(d_B[0])); // Aloca memória na GPU para a matriz B
    cudaMalloc((void**)&d_C, N * N * sizeof(d_C[0])); // Aloca memória na GPU para a matriz C
    cudaMalloc((void**)&d_D, N * N * sizeof(d_D[0])); // Aloca memória na GPU para a matriz D
    cudaMalloc((void**)&d_E, N * N * sizeof(d_E[0])); // Aloca memória na GPU para a matriz E

    // Inicializa as matrizes A e B na GPU
    init_identity<<<1, N>>>(d_A, N);
    init_unit<<<1, N>>>(d_B, N);
    init_random<<<1, N>>>(d_C, N);

    printf("Inicia Matrizes!\n"); // Imprime uma mensagem de inicio
    cublasCreate(&handle); // Inicializa o handle cuBLAS
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // Configura o modo de matemática para usar Tensor Cores
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_D, N); // Realiza a multiplicação de matrizes na GPU
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_C, N, &beta, d_E, N); // Realiza a multiplicação de matrizes na GPU

    cudaDeviceSynchronize(); // Espera a GPU terminar de executar a multiplicação de matrizes
    printf("Multiplicacao de matrizes realizada com sucesso!\n"); // Imprime uma mensagem de sucesso

    // Copia a matriz resultante para a memória do host
    double* h_D = (double*)malloc(N * N * sizeof(double));
    cudaMemcpy(h_D, d_D, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    double* h_E = (double*)malloc(N * N * sizeof(double));
    cudaMemcpy(h_E, d_E, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Imprime a matriz resultante
    printf("Matriz D:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_D[i * N + j]);
        }
        printf("\n");
    }

    // Imprime a matriz resultante
    printf("\nMatriz E:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_E[i * N + j]);
        }
        printf("\n");
    }

    // Libera a memória
    free(h_D);
    free(h_E);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset(); // Reseta o dispositivo CUDA
    return 0; // Retorna 0 indicando que o programa terminou com sucesso
}

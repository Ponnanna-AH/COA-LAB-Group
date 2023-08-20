#include <iostream>
#include <cassert>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>

const int N = 128;
const int BLOCK_SIZE = 16;

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for (int i = 0; i < n / BLOCK_SIZE; ++i)
    {
        shared_A[ty][tx] = A[row * n + i * BLOCK_SIZE + tx];
        shared_B[ty][tx] = B[(i * BLOCK_SIZE + ty) * n + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        __syncthreads();
    }

    C[row * n + col] = sum;
}

void verifyMatrixMultiplication(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            assert(fabs(C[i * n + j] - sum) < 1e-5);
        }
    }
}

int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int matrixSize = N * N * sizeof(float);

    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];

    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = static_cast<float>(rand() % 100 + 1);
        h_B[i] = static_cast<float>(rand() % 100 + 1);
    }

    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);

    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    matrixMultiplyShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    verifyMatrixMultiplication(h_A, h_B, h_C, N);

    std::cout << "Matrix multiplication result is correct." << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

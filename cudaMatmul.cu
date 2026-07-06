#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

template<int blockSize>
__global__ void Matmul(float *A, float *B, float *C, int N, int M, int K) {
	// ensure : threads(blockSize, blockSize)
	int idxx = threadIdx.x + blockSize * blockIdx.x;
	int idxy = threadIdx.y + blockSize * blockIdx.y;
	__shared__ float AsT[blockSize][blockSize], Bs[blockSize][blockSize];
	float ans = 0;
	for (int i = 0; i < K; i += blockSize) {
		if (idxx < N && threadIdx.y + i < K) {
			AsT[threadIdx.y][threadIdx.x] = A[idxx * K + threadIdx.y + i];
		}
		else AsT[threadIdx.y][threadIdx.x] = 0;
		if (threadIdx.x + i < K && idxy < M) {
			Bs[threadIdx.x][threadIdx.y] = B[(threadIdx.x + i) * M + idxy];
		}
		else Bs[threadIdx.x][threadIdx.y] = 0;
		__syncthreads();

		#pragma unroll
		for (int k = 0; k < blockSize; k++) {
			ans += AsT[threadIdx.x][k] * Bs[k][threadIdx.y];
		}
		if (i + blockSize < K) __syncthreads();
	}
	if (idxx < N && idxy < M) C[idxx * M + idxy] = ans;
}

void Matmul(int N, int M, int K) {
	float *A, *B;
	float *devA, *devB, *devC;

	CUDA_CHECK(cudaMallocHost(&A, N * K * sizeof(float)));
	CUDA_CHECK(cudaMallocHost(&B, K * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devA, N * K * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devB, K * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devC, N * M * sizeof(float)));

	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N * K; i++) A[i] = distr(rnd);
	for (int i = 0; i < K * M; i++) B[i] = distr(rnd);

	CUDA_CHECK(cudaMemcpy(devA, A, N * K * sizeof(float), cudaMemcpyDefault));
	CUDA_CHECK(cudaMemcpy(devB, B, K * M * sizeof(float), cudaMemcpyDefault));
	
	dim3 threads(32, 32);
	dim3 blocks(cuda::ceil_div(N, 32), cuda::ceil_div(M, 32));

	for (int _ = 0; _ < 3; _++) {
		Matmul<32><<<blocks, threads>>>(devA, devB, devC, N, M, K);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	auto start = chrono::high_resolution_clock::now();
	Matmul<32><<<blocks, threads>>>(devA, devB, devC, N, M, K);
	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur = end - start;
	printf("time used : %lf ms\n", dur.count());


	CUDA_CHECK(cudaFree(devA));
	CUDA_CHECK(cudaFree(devB));
	CUDA_CHECK(cudaFree(devC));
	CUDA_CHECK(cudaFreeHost(A));
	CUDA_CHECK(cudaFreeHost(B));
}

int main() {
	Matmul(1 << 12, 1 << 12, 1 << 12);
}
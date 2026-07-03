#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
using namespace std;

__global__ void vecAdd(float *A, float *B, float *C, int N) {
	/*
		threadIdx
		blockDim
		blockIdx
		gridDim
		.x .y .z (3d)
	*/

	// get the working index
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < N) C[idx] = A[idx] + B[idx];
}
void unifiedMem(int N) {
	float *A, *B, *C;
	cudaMallocManaged(&A, N * sizeof(float));
	cudaMallocManaged(&B, N * sizeof(float));
	cudaMallocManaged(&C, N * sizeof(float));
	
	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N; i++) A[i] = distr(rnd);
	for (int i = 0; i < N; i++) B[i] = distr(rnd);

	int threads = 256;
	int blocks = cuda::ceil_div(N, threads);
	vecAdd<<<blocks, threads>>>(A, B, C, N);
	// usage for >1 dim : MatAdd<<<dim3(16, 16), dim3(8, 8)>>>

	cudaDeviceSynchronize();

	for (int i = 0; i < N; i++) printf("%f%c", C[i], " \n"[i + 1 == N]);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}

void explicitMem(int N) {
	float *A, *B, *C;
	float *devA, *devB, *devC;

	cudaMallocHost(&A, N * sizeof(float));
	cudaMallocHost(&B, N * sizeof(float));
	cudaMallocHost(&C, N * sizeof(float));
	cudaMalloc(&devA, N * sizeof(float));
	cudaMalloc(&devB, N * sizeof(float));
	cudaMalloc(&devC, N * sizeof(float));

	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N; i++) A[i] = distr(rnd);
	for (int i = 0; i < N; i++) B[i] = distr(rnd);

	cudaMemcpy(devA, A, N * sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(devB, B, N * sizeof(float), cudaMemcpyDefault);
	cudaMemset(devA, 0, N * sizeof(float));

	int threads = 256;
	int blocks = cuda::ceil_div(N, threads);
	vecAdd<<<blocks, threads>>>(devA, devB, devC, N);

	cudaDeviceSynchronize();

	cudaMemcpy(C, devC, N * sizeof(float), cudaMemcpyDefault);

	for (int i = 0; i < N; i++) printf("%f%c", C[i], " \n"[i + 1 == N]);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
}

int main() {
	int N = 1024;
	unifiedMem(N);
	explicitMem(N);
}
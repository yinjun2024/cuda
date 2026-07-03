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
int main() {
	const int N = 1024;
	float *A, *B, *C;
	cudaMallocManaged(&A, N * sizeof(float));
	cudaMallocManaged(&B, N * sizeof(float));
	cudaMallocManaged(&C, N * sizeof(float));
	
	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N; i++) A[i] = distr(rnd);
	for (int i = 0; i < N; i++) B[i] = distr(rnd);

	const int threads = 256;
	const int blocks = cuda::ceil_div(N, threads);
	vecAdd<<<blocks, threads>>>(A, B, C, N);
	// usage for >1 dim : MatAdd<<<dim3(16, 16), dim3(8, 8)>>>

	cudaDeviceSynchronize();

	for (int i = 0; i < N; i++) printf("%f%c", C[i], " \n"[i + 1 == N]);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	return 0;
}
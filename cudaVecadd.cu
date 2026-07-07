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

/*
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

	// for (int i = 0; i < N; i++) printf("%f%c", C[i], " \n"[i + 1 == N]);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}
*/

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
	cudaMemset(devC, 0, N * sizeof(float));

	int threads = 256;
	int blocks = cuda::ceil_div(N, threads);

	for (int _ = 0; _ < 15; _++) {
		vecAdd<<<blocks, threads>>>(devA, devB, devC, N);
		cudaDeviceSynchronize();
	}

	auto start = chrono::high_resolution_clock::now();
	vecAdd<<<blocks, threads>>>(devA, devB, devC, N);
	cudaDeviceSynchronize();
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur = end - start;
	printf("time used : %lf ms\n", dur.count());

	cudaMemcpy(C, devC, N * sizeof(float), cudaMemcpyDefault);

	float *ans = new float[N];
	for (int i = 0; i < N; i++) ans[i] = A[i] + B[i];
	if (memcmp(ans, C, N * sizeof(float)) == 0) fprintf(stderr, "Correct!\n");
	else fprintf(stderr, "Result Mismatch!\n");
	// for (int i = 0; i < N; i++) printf("%f%c", C[i], " \n"[i + 1 == N]);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	delete[] ans;
}

/*
void cpu(int N) {
	float *A, *B, *C;
	A = new float[N];
	B = new float[N];
	C = new float[N];
	
	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N; i++) A[i] = distr(rnd);
	for (int i = 0; i < N; i++) B[i] = distr(rnd);

	for (int i = 0; i < N; i++) C[i] = A[i] + B[i];

	// for (int i = 0; i < N; i++) printf("%f%c", C[i], " \n"[i + 1 == N]);

	delete[] A;
	delete[] B;
	delete[] C;
}
*/

int main() {
	int N = 1 << 27;
	// cpu(N);
	// unifiedMem(N);
	explicitMem(N);
}
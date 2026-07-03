#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void vecAdd(float *A, float *B, float *C) {
	/*
		threadIdx
		blockDim
		blockIdx
		gridDim
		.x .y .z (3d)
	*/

	// get the working index
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	C[idx] = A[idx] + B[idx];
}
int main() {
	const int N = 1024;
	float *A, *B, *C;
	A = new float[N];
	B = new float[N];
	C = new float[N];
	
	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N; i++) A[i] = distr(rnd);
	for (int i = 0; i < N; i++) B[i] = distr(rnd);

	vecAdd<<<4, 256>>>(A, B, C);
	// usage for >1 dim : MatAdd<<<dim3(16, 16), dim3(8, 8)>>>

	for (int i = 0; i < N; i++) printf("%f%c", C[i], " \n"[i + 1 == N]);

	delete[] A;
	delete[] B;
	delete[] C;
	return 0;
}
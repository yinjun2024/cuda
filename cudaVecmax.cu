#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
using namespace std;

template<int blockSize>
__global__ void maxReduce(float *a, float *b, int n) {
	__shared__ float s[blockSize]; 

	int idx = threadIdx.x + blockSize * blockIdx.x;
	int tid = threadIdx.x;

	float val = FLT_MIN;
	for (int i = idx; i < n; i += blockSize * gridDim.x) {
		val = max(val, a[i]);
	} s[tid] = val; __syncthreads();

	#define F(x) if (blockSize >= 2 * x) {if (tid < x) s[tid] = max(s[tid], s[tid + x]); __syncthreads();}
	F(512) F(256) F(128) F(64)
	#undef F

	if (tid < 32) {
		float val = max(s[tid], s[tid + 32]);
		#define G(x) val = max(val, __shfl_down_sync(0xffffffff, val, x));
		G(16) G(8) G(4) G(2) G(1)
		#undef G
		if (tid == 0) b[blockIdx.x] = val;
	}
}

void Vecmaxreduce(int N) {
	constexpr int threads = 256;
	int blocks = cuda::ceil_div(N, 8 * threads);

	float *A;
	float *devA, *devB;

	cudaMallocHost(&A, N * sizeof(float));
	cudaMalloc(&devA, N * sizeof(float));
	cudaMalloc(&devB, cuda::ceil_div(N, threads) * sizeof(float));

	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N; i++) A[i] = distr(rnd);

	cudaMemcpy(devA, A, N * sizeof(float), cudaMemcpyDefault);

	for (int _ = 0; _ < 3; _++) {
		maxReduce<threads><<<blocks, threads>>>(devA, devB, N);
		cudaDeviceSynchronize();
	}

	auto start = chrono::high_resolution_clock::now();
	maxReduce<threads><<<blocks, threads>>>(devA, devB, N);
	cudaDeviceSynchronize();
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur = end - start;
	printf("time used : %lf ms\n", dur.count());


	cudaFree(devA);
	cudaFree(devB);
	cudaFreeHost(A);
}

int main() {
	Vecmaxreduce(1 << 27);
}
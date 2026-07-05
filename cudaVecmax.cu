#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
using namespace std;

__device__ float maxReduceWarp(float val) {
	#define G(x) val = max(val, __shfl_down_sync(0xffffffff, val, x));
	G(16) G(8) G(4) G(2) G(1)
	#undef G
	return val;
}

template<int blockSize>
__global__ void maxReduce(float *a, float *b, int n) {
	int idx = threadIdx.x + blockSize * blockIdx.x;
	int tid = threadIdx.x;
	int warpNum = cuda::ceil_div(blockSize, 32);

	float val = FLT_MIN;
	for (int i = idx; i < n; i += blockSize * gridDim.x) {
		val = max(val, __ldg(a + i));
	}
	
	val = maxReduceWarp(val);
	__shared__ float tmp[32];
	if ((tid & 0x1f) == 0) tmp[tid >> 5] = val;
	__syncthreads();
	
	if (tid < 32) {
		if (tid < warpNum) val = tmp[tid]; else val = FLT_MIN;
		val = maxReduceWarp(val);
		if (tid == 0) b[blockIdx.x] = val;
	}
}

void Vecmaxreduce(int N) {
	constexpr int threads = 256;
	int blocks = min(cuda::ceil_div(N, 8 * threads), 2560); // tesla T4

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
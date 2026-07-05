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

__device__ float maxReduceWarp(float val) {
	#define G(x) val = max(val, __shfl_down_sync(0xffffffff, val, x));
	G(16) G(8) G(4) G(2) G(1)
	#undef G
	return val;
}

template<int blockSize>
__global__ void maxReduce(float *a, float *b, int n) {
	float4 *a4 = reinterpret_cast<float4*>(a);
	int n4 = n >> 2;
	int idx = threadIdx.x + blockSize * blockIdx.x;
	int tid = threadIdx.x;
	int warpNum = cuda::ceil_div(blockSize, 32);

	float val = FLT_MIN;
	for (int i = idx; i < n4; i += blockSize * gridDim.x) {
		val = max(val, a4[i].x);
		val = max(val, a4[i].y);
		val = max(val, a4[i].z);
		val = max(val, a4[i].w);
	}
	for (int i = (n4 << 2) + idx; i < n; i += blockSize * gridDim.x) {
		val = max(val, a[i]);
	}
	
	val = maxReduceWarp(val);
	extern __shared__ float tmp[];
	if ((tid & 0x1f) == 0) tmp[tid >> 5] = val;
	__syncthreads();
	
	if (tid < 32) {
		if (tid < warpNum) val = tmp[tid]; else val = FLT_MIN;
		val = maxReduceWarp(val);
		if (tid == 0) b[blockIdx.x] = val;
	}
}

void Vecmaxreduce(int N) {
	constexpr int threads = 128;
	int blocks = min(cuda::ceil_div(N, threads), 2560 * 4); // tesla T4

	float *A;
	float *devA, *devB;

	CUDA_CHECK(cudaMallocHost(&A, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devA, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devB, cuda::ceil_div(N, threads) * sizeof(float)));

	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>();
	for (int i = 0; i < N; i++) A[i] = distr(rnd);

	CUDA_CHECK(cudaMemcpy(devA, A, N * sizeof(float), cudaMemcpyDefault));

	for (int _ = 0; _ < 3; _++) {
		maxReduce<threads><<<blocks, threads, cuda::ceil_div(threads, 32)>>>(devA, devB, N);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	auto start = chrono::high_resolution_clock::now();
	maxReduce<threads><<<blocks, threads, cuda::ceil_div(threads, 32)>>>(devA, devB, N);
	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur = end - start;
	printf("time used : %lf ms\n", dur.count());


	CUDA_CHECK(cudaFree(devA));
	CUDA_CHECK(cudaFree(devB));
	CUDA_CHECK(cudaFreeHost(A));
}

int main() {
	Vecmaxreduce(1 << 27);
}
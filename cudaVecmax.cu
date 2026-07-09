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
__global__ void maxReduce(float *a, int n) {
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
		if (tid == 0) a[blockIdx.x] = val;
	}
}

template<int threads>
void vecMax(int N) {
	float *A;
	float *devA;

	CUDA_CHECK(cudaMallocHost(&A, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devA, N * sizeof(float)));

	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>(-1, 1);
	for (int i = 0; i < N; i++) A[i] = distr(rnd);

	CUDA_CHECK(cudaMemcpy(devA, A, N * sizeof(float), cudaMemcpyDefault));

	for (int _ = 0; _ < 15; _++) {
		int M = N; while (M > 1) {
			int M2 = cuda::ceil_div(M, threads);
			int blocks = min(M2, 2560 * 4); // tesla T4
			maxReduce<threads><<<blocks, threads, cuda::ceil_div(threads, 32)>>>(devA, M);
			CUDA_CHECK(cudaDeviceSynchronize());
			M = M2;
		}
	}

	cudaEvent_t start, stop;
    float elapsedTime = 0.0;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int M = N; while (M > 1) {
		int M2 = cuda::ceil_div(M, threads);
		int blocks = min(M2, 2560 * 4); // tesla T4
		maxReduce<threads><<<blocks, threads, cuda::ceil_div(threads, 32)>>>(devA, M);
		CUDA_CHECK(cudaDeviceSynchronize());
		M = M2;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time used : %f ms\n", elapsedTime);

	float result;
	CUDA_CHECK(cudaMemcpy(&result, devA, sizeof(float), cudaMemcpyDefault));

	float ans = FLT_MIN;
	for (int i = 0; i < N; i++) ans = max(ans, A[i]);
	
	if (ans == result) fprintf(stderr, "Correct!\n");
	else fprintf(stderr, "Result Mismatch!\n");

	CUDA_CHECK(cudaFree(devA));
	CUDA_CHECK(cudaFreeHost(A));
}

int main() {
	vecMax<128>(1 << 27);
}
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

template<int BN, int BM, int BK, int BS>
__global__ void Matmul(float *A, float *B, float *C, int N, int M, int K) {
	// ensure : threads(BS * BS), blocks(ceil(N / BN), ceil(M / BM))
	// recommend : <128, 128, 8, 16>

	static_assert(BS * BS % BAy == 0);
	static_assert(BS * BS % BBx == 0);
	static_assert(BN % BS == 0);
	static_assert(BM % BS == 0);
	static_assert(BN % BAx == 0);
	static_assert(BM % BBy == 0);
	constexpr int BAy = BK, BAx = BS * BS / BAy;
	constexpr int BBx = BK, BBy = BS * BS / BBx;
	constexpr int BCx = BN / BS, BCy = BM / BS;
	constexpr int CA = BN / BAx, CB = BM / BBy;

	int Ax = threadIdx.x / BAy, Ay = threadIdx.x % BAy;
	int Bx = threadIdx.x / BBy, By = threadIdx.x % BBy;
	int Cx = threadIdx.x / BS, Cy = threadIdx.x % BS;
	Cx *= BCx; Cy *= BCy;
	int Sx = blockIdx.x * BN, Sy = blockIdx.y * BM;
	
	__shared__ float As[BN][BK + 1], Bs[BK + 1][BM]; // padding trick
	float Areg[BCx], Breg[BCy], Creg[BCx][BCy] = {0};
	
	for (int k = 0; k < K; k += BK) {
		#pragma unroll
		for (int i = 0; i < CA; i++) {
			int x = Ax + BAx * i + Sx, y = Ay + k;
			As[Ax + BAx * i][Ay] = x < N && y < K ? A[x * K + y] : 0;
		}
		#pragma unroll
		for (int i = 0; i < CB; i++) {
			int x = Bx + k, y = By + BBy * i + Sy;
			Bs[Bx][By + BBy * i] = x < K && y < M ? B[x * M + y] : 0;
		}
		__syncthreads();

		#pragma unroll
		for (int x = 0; x < BK; x++) {
			for (int _ = 0; _ < BCx; _++) Areg[_] = As[_ + Cx][x];
			for (int _ = 0; _ < BCy; _++) Breg[_] = Bs[x][_ + Cy];
			for (int i = 0; i < BCx; i++) for (int j = 0; j < BCy; j++) {
				Creg[i][j] += Areg[i] * Breg[j];
			}
		}
		__syncthreads();
	}

	for (int i = 0; i < BCx; i++) for (int j = 0; j < BCy; j++) {
		int x = i + Cx + Sx, y = j + Cy + Sy;
		if (x < N && y < M) C[x * M + y] = Creg[i][j];
	}
}

template<int BN, int BM, int BK, int BS>
void Matmul(int N, int M, int K) {
	float *A, *B, *C;
	float *devA, *devB, *devC;

	CUDA_CHECK(cudaMallocHost(&A, N * K * sizeof(float)));
	CUDA_CHECK(cudaMallocHost(&B, K * M * sizeof(float)));
	CUDA_CHECK(cudaMallocHost(&C, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devA, N * K * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devB, K * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devC, N * M * sizeof(float)));

	mt19937 rnd(123);
	auto distr = uniform_real_distribution<float>(-1, 1);
	for (int i = 0; i < N * K; i++) A[i] = distr(rnd);
	for (int i = 0; i < K * M; i++) B[i] = distr(rnd);

	CUDA_CHECK(cudaMemcpy(devA, A, N * K * sizeof(float), cudaMemcpyDefault));
	CUDA_CHECK(cudaMemcpy(devB, B, K * M * sizeof(float), cudaMemcpyDefault));
	
	dim3 blocks(cuda::ceil_div(N, BN), cuda::ceil_div(M, BM));

	for (int _ = 0; _ < 15; _++) {
		Matmul<BN, BM, BK, BS><<<blocks, BS * BS>>>(devA, devB, devC, N, M, K);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	auto start = chrono::high_resolution_clock::now();
	Matmul<BN, BM, BK, BS><<<blocks, BS * BS>>>(devA, devB, devC, N, M, K);
	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur = end - start;
	printf("time used : %lf ms\n", dur.count());

	CUDA_CHECK(cudaMemcpy(C, devC, N * M * sizeof(float), cudaMemcpyDefault));
	
	fprintf(stderr, "random check: random check a value from each row\n");
	bool cmp = 1; for (int i = 0; i < N; i++) {
		int j = uniform_int_distribution<>(0, M - 1)(rnd); float ans = 0;
		for (int k = 0; k < K; k++) ans += A[i * K + k] * B[k * M + j];
		if (fabs(C[i * M + j] - ans) / max(1.0f, fabs(ans)) > 1e-4) {
			cmp = 0;
			printf("! %d %d -> %f %f\n", i, j, C[i * M + j], ans);
			// break;
		}
	}
	if (cmp) fprintf(stderr, "Correct!\n");
	else fprintf(stderr, "Result Mismatch!\n");

	CUDA_CHECK(cudaFree(devA));
	CUDA_CHECK(cudaFree(devB));
	CUDA_CHECK(cudaFree(devC));
	CUDA_CHECK(cudaFreeHost(A));
	CUDA_CHECK(cudaFreeHost(B));
	CUDA_CHECK(cudaFreeHost(C));
}

int main() {
	const int S = 1 << 13;
	Matmul<128, 128, 8, 16>(S, S, S);
	// Matmul<64, 64, 16, 16>(1 << 13, 1 << 13, 1 << 13);
	// Matmul<64, 64, 16, 32>(1 << 13, 1 << 13, 1 << 13);
}
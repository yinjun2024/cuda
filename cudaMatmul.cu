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

__global__ void Matmul(float *A, float *B, float *C, int N, int M, int K) {

	// restriction : N, M, K must aligned with 4

	__shared__ float AsT[8][128], Bs[8][128 + 4];
	float Areg[8], Breg[8], Creg[8][8] = {0}; float4 val; int x, y;
	int Ap = blockIdx.y << 7, Ax = threadIdx.x >> 7, Ay = threadIdx.x & 127;
	int Bp = blockIdx.x << 7, Bx = threadIdx.x >> 3, By = threadIdx.x & 7;
	int Cx = (threadIdx.x >> 4) << 3, Cy = (threadIdx.x & 15) << 3;
	for (int k = 0; k < K; k += 8) {
		x = Ax << 2 | k, y = Ap | Ay;
		if (y < N && x < K) {
			val = reinterpret_cast<float4*>(A + y * K + x)[0];
			AsT[Ax << 2 | 0][Ay] = val.x;
			AsT[Ax << 2 | 1][Ay] = val.y;
			AsT[Ax << 2 | 2][Ay] = val.z;
			AsT[Ax << 2 | 3][Ay] = val.w;
		}
		else {
			AsT[Ax << 2 | 0][Ay] = 0;
			AsT[Ax << 2 | 1][Ay] = 0;
			AsT[Ax << 2 | 2][Ay] = 0;
			AsT[Ax << 2 | 3][Ay] = 0;
		}
		x = Bp | Bx << 2, y = By | k;
		if (y < K && x < M) {
			val = reinterpret_cast<float4*>(B + y * M + x)[0];
			Bs[By][Bx << 2 | 0] = val.x;
			Bs[By][Bx << 2 | 1] = val.y;
			Bs[By][Bx << 2 | 2] = val.z;
			Bs[By][Bx << 2 | 3] = val.w;
		}
		else {
			Bs[By][Bx << 2 | 0] = 0;
			Bs[By][Bx << 2 | 1] = 0;
			Bs[By][Bx << 2 | 2] = 0;
			Bs[By][Bx << 2 | 3] = 0;
		}
		__syncthreads();

		// #pragma unroll
		for (int k_ = 0; k_ < 8; k_++) {
			val = reinterpret_cast<float4*>(AsT[k_] + Cx)[0];
			Areg[0] = val.x;
			Areg[1] = val.y;
			Areg[2] = val.z;
			Areg[3] = val.w;
			val = reinterpret_cast<float4*>(AsT[k_] + Cx)[1];
			Areg[4] = val.x;
			Areg[5] = val.y;
			Areg[6] = val.z;
			Areg[7] = val.w;
			val = reinterpret_cast<float4*>(Bs[k_] + Cy)[0];
			Breg[0] = val.x;
			Breg[1] = val.y;
			Breg[2] = val.z;
			Breg[3] = val.w;
			val = reinterpret_cast<float4*>(Bs[k_] + Cy)[1];
			Breg[4] = val.x;
			Breg[5] = val.y;
			Breg[6] = val.z;
			Breg[7] = val.w;
			for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
				Creg[i][j] += Areg[i] * Breg[j];
			}
		}
		__syncthreads();
	}

	for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
		int x = Ap | Cx | i, y = Bp | Cy | j;
		if (x < N && y < M) C[x * M + y] = Creg[i][j];
	}
}

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
	
	dim3 blocks(cuda::ceil_div(N, 128), cuda::ceil_div(M, 128));

	for (int _ = 0; _ < 15; _++) {
		Matmul<<<blocks, 256>>>(devA, devB, devC, N, M, K);
	}
	
	CUDA_CHECK(cudaDeviceSynchronize());
	auto start = chrono::high_resolution_clock::now();
	Matmul<<<blocks, 256>>>(devA, devB, devC, N, M, K);
	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur = end - start;
	printf("time used : %lf ms\n", dur.count());

	CUDA_CHECK(cudaMemcpy(C, devC, N * M * sizeof(float), cudaMemcpyDefault));
	
	/* fprintf(stderr, "random check: random check a value from each row\n");
	bool cmp = 1; for (int i = 0; i < N; i++) {
		int j = uniform_int_distribution<>(0, M - 1)(rnd); double ans = 0;
		for (int k = 0; k < K; k++) ans += (double)A[i * K + k] * B[k * M + j];
		if (fabs(C[i * M + j] - ans) / max(1.0f, fabs(ans)) > 1e-3) {
			cmp = 0;
			printf("! %d %d -> %f %f\n", i, j, C[i * M + j], ans);
			// break;
		}
	}
	if (cmp) fprintf(stderr, "Correct!\n");
	else fprintf(stderr, "Result Mismatch!\n"); */

	CUDA_CHECK(cudaFree(devA));
	CUDA_CHECK(cudaFree(devB));
	CUDA_CHECK(cudaFree(devC));
	CUDA_CHECK(cudaFreeHost(A));
	CUDA_CHECK(cudaFreeHost(B));
	CUDA_CHECK(cudaFreeHost(C));
}

int main() {
	const int S = 1 << 13;
	Matmul(S, S, S);
	// Matmul<64, 64, 16, 16>(1 << 13, 1 << 13, 1 << 13);
	// Matmul<64, 64, 16, 32>(1 << 13, 1 << 13, 1 << 13);
}
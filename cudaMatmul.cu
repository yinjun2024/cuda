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

__global__ void Matmul_sync(float *A, float *B, float *C, int N, int M, int K) {
    // blocks(ceil_div(N, 128), ceil_div(M, 128)), threads(256)
    // restriction : N, M, K must aligned with 128 !!!

    __shared__ float AsT[8][128], Bs[8][128];
    float Areg[8], Breg[8], Creg[8][8] = {0}; float4 val; int x, y;
    int Apos = blockIdx.y << 7, Ax = threadIdx.x >> 7, Ay = threadIdx.x & 127;
    int Bpos = blockIdx.x << 7, Bx = threadIdx.x & 31, By = threadIdx.x >> 5;
    int Warp = threadIdx.x >> 5, Lane = threadIdx.x & 31;
    int Cx = (Warp & 1) << 3 | Lane >> 3 << 1 | Lane & 1;
    int Cy = Warp >> 1 << 2 | (Lane >> 1 & 3); // Z-Shape
    for (int k = 0; k < K; k += 8) {
        x = k | Ax << 2, y = Apos | Ay;
        val = reinterpret_cast<float4*>(A + y * K + x)[0];
        AsT[Ax << 2 | 0][Ay] = val.x;
        AsT[Ax << 2 | 1][Ay] = val.y;
        AsT[Ax << 2 | 2][Ay] = val.z;
        AsT[Ax << 2 | 3][Ay] = val.w;
        x = Bpos | Bx << 2, y = k | By;
        reinterpret_cast<float4*>(Bs[By] + (Bx << 2))[0] = reinterpret_cast<float4*>(B + y * M + x)[0];
        __syncthreads();

        #pragma unroll
        for (int k_ = 0; k_ < 8; k_++) {
            reinterpret_cast<float4*>(Areg)[0] = reinterpret_cast<float4*>(AsT[k_] + (Cx << 2))[0];
            reinterpret_cast<float4*>(Areg)[1] = reinterpret_cast<float4*>(AsT[k_] + (Cx << 2 | 64))[0];
            reinterpret_cast<float4*>(Breg)[0] = reinterpret_cast<float4*>(Bs[k_] + (Cy << 2))[0];
            reinterpret_cast<float4*>(Breg)[1] = reinterpret_cast<float4*>(Bs[k_] + (Cy << 2 | 64))[0];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    Creg[i][j] += Areg[i] * Breg[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
        int x = Apos | i >> 2 << 6 | Cx << 2 | (i & 3), y = Bpos | j >> 2 << 6 | Cy << 2 | (j & 3);
        C[x * M + y] = Creg[i][j];
    }
}

__global__ void Matmul_async(float *A, float *B, float *C, int N, int M, int K) {
    // blocks(ceil_div(N, 128), ceil_div(M, 128)), threads(256)
    // restriction : N, M, K must aligned with 128 !!!

    __shared__ float AsT[2][8][128], Bs[2][8][128];
    float Areg[8], Breg[8], Creg[8][8] = {0}; float4 valA, valB; int x, y;
    int Apos = blockIdx.y << 7, Ax = threadIdx.x >> 7, Ay = threadIdx.x & 127;
    int Bpos = blockIdx.x << 7, Bx = threadIdx.x & 31, By = threadIdx.x >> 5;
    int Warp = threadIdx.x >> 5, Lane = threadIdx.x & 31;
    int Cx = (Warp & 1) << 3 | Lane >> 3 << 1 | Lane & 1;
    int Cy = Warp >> 1 << 2 | (Lane >> 1 & 3); // Z-Shape

	int write = 0, read = 1;
	x = Ax << 2, y = Apos | Ay;
	valA = reinterpret_cast<float4*>(A + y * K + x)[0];
    x = Bpos | Bx << 2, y = By;
	valB = reinterpret_cast<float4*>(B + y * M + x)[0];
	AsT[write][Ax << 2 | 0][Ay] = valA.x;
	AsT[write][Ax << 2 | 1][Ay] = valA.y;
	AsT[write][Ax << 2 | 2][Ay] = valA.z;
	AsT[write][Ax << 2 | 3][Ay] = valA.w;
	reinterpret_cast<float4*>(Bs[write][By] + (Bx << 2))[0] = valB;
	__syncthreads(); write ^= 1; read ^= 1;

    for (int k = 8; k < K; k += 8) {
		x = k | Ax << 2, y = Apos | Ay;
		valA = reinterpret_cast<float4*>(A + y * K + x)[0];
		x = Bpos | Bx << 2, y = k | By;
		valB = reinterpret_cast<float4*>(B + y * M + x)[0];

        #pragma unroll
        for (int k_ = 0; k_ < 8; k_++) {
            reinterpret_cast<float4*>(Areg)[0] = reinterpret_cast<float4*>(AsT[read][k_] + (Cx << 2))[0];
            reinterpret_cast<float4*>(Areg)[1] = reinterpret_cast<float4*>(AsT[read][k_] + (Cx << 2 | 64))[0];
            reinterpret_cast<float4*>(Breg)[0] = reinterpret_cast<float4*>(Bs[read][k_] + (Cy << 2))[0];
            reinterpret_cast<float4*>(Breg)[1] = reinterpret_cast<float4*>(Bs[read][k_] + (Cy << 2 | 64))[0];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    Creg[i][j] += Areg[i] * Breg[j];
                }
            }
        }

		AsT[write][Ax << 2 | 0][Ay] = valA.x;
		AsT[write][Ax << 2 | 1][Ay] = valA.y;
		AsT[write][Ax << 2 | 2][Ay] = valA.z;
		AsT[write][Ax << 2 | 3][Ay] = valA.w;
		reinterpret_cast<float4*>(Bs[write][By] + (Bx << 2))[0] = valB;
        __syncthreads(); write ^= 1; read ^= 1;
    }

    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
        int x = Apos | i >> 2 << 6 | Cx << 2 | (i & 3), y = Bpos | j >> 2 << 6 | Cy << 2 | (j & 3);
        C[x * M + y] = Creg[i][j];
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
	for (int i = 0; i < N * K; i++) A[i] = i == K + 1 ? 1 : 0;//distr(rnd);
	for (int i = 0; i < K * M; i++) B[i] = i == 5 ? 1 : 0;//distr(rnd);

	CUDA_CHECK(cudaMemcpy(devA, A, N * K * sizeof(float), cudaMemcpyDefault));
	CUDA_CHECK(cudaMemcpy(devB, B, K * M * sizeof(float), cudaMemcpyDefault));
	
	dim3 blocks(cuda::ceil_div(N, 128), cuda::ceil_div(M, 128));

	for (int _ = 0; _ < 15; _++) {
		Matmul_async<<<blocks, 256>>>(devA, devB, devC, N, M, K);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	
	cudaEvent_t start, stop;
    float elapsedTime = 0.0;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	Matmul_async<<<blocks, 256>>>(devA, devB, devC, N, M, K);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time used : %f ms\n", elapsedTime);

	CUDA_CHECK(cudaMemcpy(C, devC, N * M * sizeof(float), cudaMemcpyDefault));
	
	bool cmp = 1; for (int i = 0; i < N; i++) {
		// int j = uniform_int_distribution<>(0, M - 1)(rnd); {
		for (int j = 0; j < M; j++) {
			double ans = 0;
			for (int k = 0; k < K; k++) ans += (double)A[i * K + k] * B[k * M + j];
			if (fabs(C[i * M + j] - ans) / max(1.0f, fabs(ans)) > 1e-3) {
				cmp = 0;
				printf("! %d %d -> %f %f\n", i, j, C[i * M + j], ans);
				// break;
			}
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
	const int S = 1 << 7;
	Matmul(S, S, S);
	// Matmul<64, 64, 16, 16>(1 << 13, 1 << 13, 1 << 13);
	// Matmul<64, 64, 16, 32>(1 << 13, 1 << 13, 1 << 13);
}
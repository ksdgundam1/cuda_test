#include <stdio.h>
#include <cstdlib>
#include <ctime>

#define SIZE 5

__global__ void addGPU(int* d, const int* a, const int* b, const int* c)
{
    int i = threadIdx.x;
    d[i] = a[i] + b[i] + c[i];
}

int main()
{
    srand((unsigned int)time(NULL));

    int a[SIZE], b[SIZE], c[SIZE];

    for (int i = 0; i < SIZE; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 100;
        c[i] = rand() % 1000;
    }

    int d[SIZE] = { 0 };

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    int* dev_d = 0;

    cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_d, SIZE * sizeof(int));

    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    addGPU << <1, SIZE >> > (dev_d, dev_a, dev_b, dev_c);
    //cudaDeviceSynchronize();

    cudaMemcpy(d, dev_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("   {%d, %d, %d, %d, %d}\n + {%d, %d, %d, %d, %d}\n + {%d, %d, %d, %d, %d}\n = {%d, %d, %d, %d, %d}\n", a[0], a[1], a[2], a[3], a[4],
        b[0], b[1], b[2], b[3], b[4],
        c[0], c[1], c[2], c[3], c[4],
        d[0], d[1], d[2], d[3], d[4]);

    cudaFree(dev_a);      //Device에서 동적할당받은 메모리 반환
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    return 0;
}
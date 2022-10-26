// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define N 8
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaracion
	float *hstA_matriz;
	float *dev_matrizA;
	float *hstB_matriz;
	float *dev_matrizB;

	// reserva en el host
	hstA_matriz = (float*)malloc(N * sizeof(float));
	hstB_matriz = (float*)malloc(N * sizeof(float));

	// reserva en el device
	cudaMalloc((void**)&dev_matrizA, N * sizeof(float));
	cudaMalloc((void**)&dev_matrizB, N * sizeof(float));

	// inicializacion de datos en el host
	srand((int)time(NULL));
	for (int i = 0; i<N; i++)
	{
		hstA_matriz[i] = (float)rand() / RAND_MAX; 

	}
	
	//visualizacion de datos en el host
	printf("DATOS HOST A:\n");
	for (int i = 0; i<N; i++)
	{
		printf("A[%i] = %.2f\n", i, hstA_matriz[i]);
	}


	// copia de datos CPU -> GPU
	cudaMemcpy(dev_matrizA, hstA_matriz, N * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_matrizB, dev_matrizA, N * sizeof(float), cudaMemcpyDeviceToDevice);

	cudaMemcpy(hstB_matriz, dev_matrizB, N * sizeof(float), cudaMemcpyDeviceToHost);



	//visualizacion de datos en el host
	printf("\n\nDATOS HOST B:\n");
	for (int i = 0; i<N; i++)
	{
		printf("A[%i] = %.2f\n", i, hstB_matriz[i]);
	}

	
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_launch_parameters.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernel.h"

int vecLong = 0;


// declaracion de funciones
__host__ void impares_CPU(int *hst_impares)
{
	for (int i = 0; i < vecLong; i++)
	{
		
		hst_impares[i] = rand() % 9;
	}
}
__global__ void suma_GPU(int *vector_1, int *vector_2, int *vector_suma, int n)
{
		int id = threadIdx.x;

		int pos = ((n - 1) - id);

		vector_2[id] = vector_1[pos];



		vector_suma[id] = vector_1[id] + vector_2[id];
}

int main(int argc, char** argv)
{
	

	printf("Longitud de los vectores : ");


	scanf("%i", &vecLong);

	getchar();


	int *hst_vector1, *hst_vector2, *hst_resultado;
	int *dev_vector1, *dev_vector2, *dev_resultado;

	hst_vector1 = (int*)malloc(vecLong * sizeof(int));
	hst_vector2 = (int*)malloc(vecLong * sizeof(int));
	hst_resultado = (int*)malloc(vecLong * sizeof(int));

	cudaMalloc((void**)&dev_vector1, vecLong * sizeof(int));
	cudaMalloc((void**)&dev_vector2, vecLong * sizeof(int));
	cudaMalloc((void**)&dev_resultado, vecLong * sizeof(int));

	printf("Generamos el vector 1\n");	
	impares_CPU(hst_vector1);




	cudaMemcpy(dev_vector1, hst_vector1, vecLong * sizeof(int), cudaMemcpyHostToDevice);

	printf("Generamos el vector 2 y sumamos\n");
	suma_GPU <<<1,vecLong>>>(dev_vector1, dev_vector2, dev_resultado, vecLong);




	cudaMemcpy(hst_vector2, dev_vector2, vecLong * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_resultado, dev_resultado, vecLong * sizeof(int), cudaMemcpyDeviceToHost);

	printf("> RESULTADOS:\n");
	printf("VECTOR 1:\n");
	for (int i = 0; i < vecLong; i++)
	{
		printf("%2d ", hst_vector1[i]);
	}
	printf("\n");
	printf("VECTOR 2:\n");
	for (int i = 0; i < vecLong; i++)
	{
		printf("%2d ", hst_vector2[i]);
	}
	printf("\n");
	printf("SUMA:\n");
	for (int i = 0; i < vecLong; i++)
	{
		printf("%2d ", hst_resultado[i]);
	}
	printf("\n");
	printf("***************************************************\n");
	printf("Pulsa [INTRO] para finalizar");
	getchar();
	return 0;
}
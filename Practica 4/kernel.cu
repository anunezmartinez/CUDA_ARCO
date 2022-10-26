// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
// defines
#define TAM 10 // tamaño de la matriz TAMxTAM
#define TAM1 22 // tamaño de la matriz TAMxTAM
// declaracion de funciones
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void suma_gpu(float *A, float *B, float *C)
{
	// indice de fila
	int miFila = threadIdx.y;
	// indice de columna
	int miColumna = threadIdx.x;
	// Calculamos la suma:
	// C[fila][columna] = A[fila][columna] + B[fila][columna]
	// Para ello convertimos los indices de 'fila' y 'columna' en un indice lineal
	int myID = miColumna + miFila * blockDim.x;
	// sumamos cada elemento
	C[myID] = A[myID] + B[myID];
}

__host__ void properties_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char *archName;
	switch (major)
	{
	case 1:

		archName = "TESLA";
		cudaCores = 8;
		break;
	case 2:

		archName = "FERMI";
		if (minor == 0)
			cudaCores = 32;
		else
			cudaCores = 48;
		break;
	case 3:

		archName = "KEPLER";
		cudaCores = 192;
		break;

	case 4:

		archName = "MAXWELL";
		cudaCores = 128;
		break;
	case 5:

		archName = "PASCAL";
		cudaCores = 64;
		break;
	case 6:
		cudaCores = 64;
		if (minor == 0)
			archName = "VOLTA";
		else
			archName = "TURING";
		break;
	case 7:
		archName = "AMPERE";
		cudaCores = 8;
		break;



	default:
		archName = "UNKNOWN";
	}
	int rtV;
	cudaRuntimeGetVersion(&rtV);

	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit \t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> CUDA Architecture \t: %s\n", archName);
	printf("> Computing Capacity \t: %d.%d\n", major, minor);
	printf("> No. of MultiProcessors \t: %d\n", SM);
	printf("> No. CUDA cores (%dx%d) \t: %d\n", cudaCores, SM, cudaCores*SM);
	printf("> Global Memory (total) \t: %u MiB\n",
		deviceProp.totalGlobalMem / (1024 * 1024));
	printf("***************************************************\n");
}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{

	// declaraciones
	float *hst_A, *hst_B, *hst_C;
	float *dev_A, *dev_B, *dev_C;
	// reserva en el host
	hst_A = (float*)malloc(TAM*TAM1 * sizeof(float));
	hst_B = (float*)malloc(TAM*TAM1 * sizeof(float));
	hst_C = (float*)malloc(TAM*TAM1 * sizeof(float));
	// reserva en el device
	cudaMalloc((void**)&dev_A, TAM*TAM1 * sizeof(float));
	cudaMalloc((void**)&dev_B, TAM*TAM1 * sizeof(float));
	cudaMalloc((void**)&dev_C, TAM*TAM1 * sizeof(float));
	// incializacion
	
	srand((int)time(NULL));


	for (int i = 0; i<TAM*TAM1; i++)
	{
		hst_A[i] = (int)(rand() % 9) + 1;
		
		if (i % 2 != 0) {
			hst_B[i] = 0;
			//hst_B[i] = (float)rand() / RAND_MAX;
		}
		else {
			hst_B[i] = (int)(rand() % 9) + 1;
		}

		
	
	}
	// copia de datos
	cudaMemcpy(dev_A, hst_A, TAM*TAM1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, hst_B, TAM*TAM1 * sizeof(float), cudaMemcpyHostToDevice);
	// dimensiones del kernel
	dim3 Nbloques(1);
	dim3 hilosB(TAM, TAM1);
	// llamada al kernel bidimensional de NxN hilos
	suma_gpu << <Nbloques, hilosB >> >(dev_A, dev_B, dev_C);
	// recogida de datos
	cudaMemcpy(hst_C, dev_C, TAM*TAM1 * sizeof(float), cudaMemcpyDeviceToHost);
	// impresion de resultados

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0)
	{
		printf("!!NO CUDA DEVICE DETECTED!!\n");
		printf("<hit [ENTER] to exit>");
		getchar();
		return 1;
	}
	else
	{
		printf("CUDA devices found: <%d> \n", deviceCount);
		for (int id = 0; id < deviceCount; id++)
		{
			properties_Device(id);
		}
	}


	printf("Número de hilos : %d \n", TAM * TAM1);
	printf("Número de bloques : %d \n", Nbloques);
	printf("Tamaño eje X : %d \n", TAM);
	printf("Tamaño eje Y : %d \n", TAM1);
	
	printf("MATRIZ A:\n");
	for (int i = 0; i<TAM; i++)
	{
		for (int j = 0; j<TAM1; j++)
		{
			printf("%.0f ", hst_A[j + i*TAM]);
		}
		printf("\n");
	}
	printf("MATRIZ B:\n");
	for (int i = 0; i<TAM; i++)
	{
		for (int j = 0; j<TAM1; j++)
		{
			printf("%.0f ", hst_B[j + i*TAM]);
		}
		printf("\n");
	}

	
	// salida
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}





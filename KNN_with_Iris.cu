
//Pia Wetzel



/*
Program uses the KNN (with K = 3) classification algorithm to classify the Iris species Setosa, Virginica, and Versicolor. 
Goal is it to identify an Iris species based on the four parameters Sepal-width, Sepal-length, Petal-width, and Petal-length.
*/



#include <stdio.h>
#include <cuda.h>
#include "math.h"

#include<stdio.h>
#include<string.h>



__global__ void knn (double *oMatrix, double *topN, double*knns, unsigned matrixsize) {


  //Extracts the species of the "k" best euclidean distances
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	for(unsigned i = 0 ; i < matrixsize-1; i++)
	{
	 if(oMatrix[i] == topN[id]){
	 knns[id] = oMatrix[i+1];
	}
	}

}

//Calculated Euclidean Distance between each matrix row and a given test vector. 
//The "result" is a matrix with x rows and 2 colums, containing the euclidean 
//distance of the row plus the numerical idenifier of the Iris species associated with the row
__global__ void eucl_dist ( double *matrix, double *test, double *result, double *result2, unsigned matrixsize) {

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned max = 4;

  for (unsigned jj = 0; jj < max; ++jj) {
		result[id*2] += (matrix[id * matrixsize+jj] - test[jj])*(matrix[id * matrixsize +jj] - test[jj]);
		result2[id] += (matrix[id * matrixsize+jj] - test[jj])*(matrix[id * matrixsize +jj] - test[jj]);

	if(jj == 0)
	{
		result[id*2 + 1] = matrix[id * matrixsize + 4];
	}
	if(jj == max-1)
	{
		result[id*2] = sqrt(result[2*id]);
		result2[id] = sqrt(result2[id]);
	}

  }
}


//Parallel sorts the matrix with x rows and 2 column, ordered by increasing euclidean distance
__global__ void even_sort(double *arr, unsigned matrixsize) {

	double temp;
	int id = (threadIdx.x)*2;

	if(id <= matrixsize-2)
	{
	if(arr[id] > arr[id+1])
	{
		temp = arr[id];
		arr[id] = arr[id+1];
		arr[id+1] = temp;
	}
	}
}

__global__ void odd_sort(double *arr, unsigned matrixsize) {
	double temp;
	int id = (threadIdx.x)*2+1;

	if(id <= matrixsize-2)
	{
	if(arr[id] > arr[id+1])
	{
		temp = arr[id];
		arr[id] = arr[id+1];
		arr[id+1] = temp;
	}
	}
}



	#define N 147
	#define M 5
int main() {


	const unsigned KNN = 3;

	//Some test values ("randomly" taken out of original data set)

	//double test_iris[5] = {5.6,3.0,4.5,1.5,200};  //I'm a Versicolor
	double test_iris[5] = {5.1,3.5,1.4,0.2, 100}; //I'm a Setosa
	//double test_iris[5] = {6.7,3.0,5.2,2.3,300}; //I'm a Virginica

	dim3 block(N, M, 1);

	double *eucl_distance,*eucl_distance2, *result,*result2, *test, *matrix, *knns, *knnres;

	cudaMalloc(&result,2*N*sizeof(double));
	cudaMalloc(&result2,N*sizeof(double));

	cudaMalloc(&test,(M)*sizeof(double));
	cudaMalloc(&matrix,N*M*sizeof(double));
	cudaMalloc(&knnres,KNN*sizeof(double));

	eucl_distance = (double *)malloc(2*N * sizeof(double));
	eucl_distance2 = (double *)malloc(N * sizeof(double));
	knns = (double *)malloc(KNN * sizeof(double));


	  //Training data

	//Setosa = 100
	//Versicolor = 200
	//Virginica = 300

	//Data is taken from https://archive.ics.uci.edu/ml/datasets/iris
	double iris2[147][5] ={
	{4.9,3.0,1.4,0.2,100},
	{4.7,3.2,1.3,0.2,100},
	{4.6,3.1,1.5,0.2,100},
	{5.0,3.6,1.4,0.2,100},
	{5.4,3.9,1.7,0.4,100},
	{4.6,3.4,1.4,0.3,100},
	{5.0,3.4,1.5,0.2,100},
	{4.4,2.9,1.4,0.2,100},
	{4.9,3.1,1.5,0.1,100},
	{5.4,3.7,1.5,0.2,100},
	{4.8,3.4,1.6,0.2,100},
	{4.8,3.0,1.4,0.1,100},
	{4.3,3.0,1.1,0.1,100},
	{5.8,4.0,1.2,0.2,100},
	{5.7,4.4,1.5,0.4,100},
	{5.4,3.9,1.3,0.4,100},
	{5.1,3.5,1.4,0.3,100},
	{5.7,3.8,1.7,0.3,100},
	{5.1,3.8,1.5,0.3,100},
	{5.4,3.4,1.7,0.2,100},
	{5.1,3.7,1.5,0.4,100},
	{4.6,3.6,1.0,0.2,100},
	{5.1,3.3,1.7,0.5,100},
	{4.8,3.4,1.9,0.2,100},
	{5.0,3.0,1.6,0.2,100},
	{5.0,3.4,1.6,0.4,100},
	{5.2,3.5,1.5,0.2,100},
	{5.2,3.4,1.4,0.2,100},
	{4.7,3.2,1.6,0.2,100},
	{4.8,3.1,1.6,0.2,100},
	{5.4,3.4,1.5,0.4,100},
	{5.2,4.1,1.5,0.1,100},
	{5.5,4.2,1.4,0.2,100},
	{4.9,3.1,1.5,0.1,100},
	{5.0,3.2,1.2,0.2,100},
	{5.5,3.5,1.3,0.2,100},
	{4.9,3.1,1.5,0.1,100},
	{4.4,3.0,1.3,0.2,100},
	{5.1,3.4,1.5,0.2,100},
	{5.0,3.5,1.3,0.3,100},
	{4.5,2.3,1.3,0.3,100},
	{4.4,3.2,1.3,0.2,100},
	{5.0,3.5,1.6,0.6,100},
	{5.1,3.8,1.9,0.4,100},
	{4.8,3.0,1.4,0.3,100},
	{5.1,3.8,1.6,0.2,100},
	{4.6,3.2,1.4,0.2,100},
	{5.3,3.7,1.5,0.2,100},
	{5.0,3.3,1.4,0.2,100},
	{7.0,3.2,4.7,1.4,200},
	{6.4,3.2,4.5,1.5,200},
	{6.9,3.1,4.9,1.5,200},
	{5.5,2.3,4.0,1.3,200},
	{6.5,2.8,4.6,1.5,200},
	{5.7,2.8,4.5,1.3,200},
	{6.3,3.3,4.7,1.6,200},
	{4.9,2.4,3.3,1.0,200},
	{6.6,2.9,4.6,1.3,200},
	{5.2,2.7,3.9,1.4,200},
	{5.0,2.0,3.5,1.0,200},
	{5.9,3.0,4.2,1.5,200},
	{6.0,2.2,4.0,1.0,200},
	{6.1,2.9,4.7,1.4,200},
	{5.6,2.9,3.6,1.3,200},
	{6.7,3.1,4.4,1.4,200},
	{5.8,2.7,4.1,1.0,200},
	{6.2,2.2,4.5,1.5,200},
	{5.6,2.5,3.9,1.1,200},
	{5.9,3.2,4.8,1.8,200},
	{6.1,2.8,4.0,1.3,200},
	{6.3,2.5,4.9,1.5,200},
	{6.1,2.8,4.7,1.2,200},
	{6.4,2.9,4.3,1.3,200},
	{6.6,3.0,4.4,1.4,200},
	{6.8,2.8,4.8,1.4,200},
	{6.7,3.0,5.0,1.7,200},
	{6.0,2.9,4.5,1.5,200},
	{5.7,2.6,3.5,1.0,200},
	{5.5,2.4,3.8,1.1,200},
	{5.5,2.4,3.7,1.0,200},
	{5.8,2.7,3.9,1.2,200},
	{6.0,2.7,5.1,1.6,200},
	{5.4,3.0,4.5,1.5,200},
	{6.0,3.4,4.5,1.6,200},
	{6.7,3.1,4.7,1.5,200},
	{6.3,2.3,4.4,1.3,200},
	{5.6,3.0,4.1,1.3,200},
	{5.5,2.5,4.0,1.3,200},
	{5.5,2.6,4.4,1.2,200},
	{6.1,3.0,4.6,1.4,200},
	{5.8,2.6,4.0,1.2,200},
	{5.0,2.3,3.3,1.0,200},
	{5.6,2.7,4.2,1.3,200},
	{5.7,3.0,4.2,1.2,200},
	{5.7,2.9,4.2,1.3,200},
	{6.2,2.9,4.3,1.3,200},
	{5.1,2.5,3.0,1.1,200},
	{5.7,2.8,4.1,1.3,200},
	{6.3,3.3,6.0,2.5,300},
	{5.8,2.7,5.1,1.9,300},
	{7.1,3.0,5.9,2.1,300},
	{6.3,2.9,5.6,1.8,300},
	{6.5,3.0,5.8,2.2,300},
	{7.6,3.0,6.6,2.1,300},
	{4.9,2.5,4.5,1.7,300},
	{7.3,2.9,6.3,1.8,300},
	{6.7,2.5,5.8,1.8,300},
	{7.2,3.6,6.1,2.5,300},
	{6.5,3.2,5.1,2.0,300},
	{6.4,2.7,5.3,1.9,300},
	{6.8,3.0,5.5,2.1,300},
	{5.7,2.5,5.0,2.0,300},
	{5.8,2.8,5.1,2.4,300},
	{6.4,3.2,5.3,2.3,300},
	{6.5,3.0,5.5,1.8,300},
	{7.7,3.8,6.7,2.2,300},
	{7.7,2.6,6.9,2.3,300},
	{6.0,2.2,5.0,1.5,300},
	{6.9,3.2,5.7,2.3,300},
	{5.6,2.8,4.9,2.0,300},
	{7.7,2.8,6.7,2.0,300},
	{6.3,2.7,4.9,1.8,300},
	{6.7,3.3,5.7,2.1,300},
	{7.2,3.2,6.0,1.8,300},
	{6.2,2.8,4.8,1.8,300},
	{6.1,3.0,4.9,1.8,300},
	{6.4,2.8,5.6,2.1,300},
	{7.2,3.0,5.8,1.6,300},
	{7.4,2.8,6.1,1.9,300},
	{7.9,3.8,6.4,2.0,300},
	{6.4,2.8,5.6,2.2,300},
	{6.3,2.8,5.1,1.5,300},
	{6.1,2.6,5.6,1.4,300},
	{7.7,3.0,6.1,2.3,300},
	{6.3,3.4,5.6,2.4,300},
	{6.4,3.1,5.5,1.8,300},
	{6.0,3.0,4.8,1.8,300},
	{6.9,3.1,5.4,2.1,300},
	{6.7,3.1,5.6,2.4,300},
	{6.9,3.1,5.1,2.3,300},
	{5.8,2.7,5.1,1.9,300},
	{6.8,3.2,5.9,2.3,300},
	{6.7,3.3,5.7,2.5,300},
	{6.3,2.5,5.0,1.9,300},
	{6.5,3.0,5.2,2.0,300},
	{6.2,3.4,5.4,2.3,300},
	{5.9,3.0,5.1,1.8,300}};




	cudaMemcpy(matrix,iris2, N*M*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(test,test_iris, (M)*sizeof(double), cudaMemcpyHostToDevice);

	eucl_dist<<<1, N>>>(matrix, test, result,result2, M);
	cudaMemcpy(eucl_distance, result, 2*N * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(eucl_distance2, result2, N * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(result2,eucl_distance2, N*sizeof(double), cudaMemcpyHostToDevice);
		
		for(unsigned i = 0; i <= N/2; i++){

			even_sort<<<1, N>>>(result2, N);
			odd_sort<<<1, N>>>(result2, N);
		}

	cudaMemcpy(eucl_distance2, result2, N*sizeof(double), cudaMemcpyDeviceToHost);


	cudaMemcpy(result,eucl_distance, 2*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(result2,eucl_distance2, N*sizeof(double), cudaMemcpyHostToDevice);
	knn<<<1,3>>>(result, result2, knnres, 2*N);

	cudaMemcpy(knns, knnres, KNN*sizeof(double), cudaMemcpyDeviceToHost);
	unsigned versicolor, virginica, setosa;



	   for (unsigned i = 0; i < KNN; ++i) {
		  if(knns[i] == 100){setosa++;}
		  else if(knns[i] == 200){versicolor++;}
		  else if (knns[i] == 300){virginica++;}
	    }

	    printf("\n--------------------------------------------------------\n");
	    printf("\n\n\nInput:\n\nSepal-length: %2f\nSepal-width: %2f\nPetal-length: %2f\nPetal-width: %2f", test_iris[0], test_iris[1], test_iris[2], test_iris[3]);
	    printf("\n\nThe %2d closest neighbors:\nsetosa: %2d virginica: %2d versicolor: %2d",KNN, setosa, virginica, versicolor);
	    printf("\n\nApplying KNN classification with k=%2d yields: ", KNN);

	    if(setosa > virginica && setosa > versicolor)
	    {
	      printf("The input is a Setosa\n\n");
	    }else if(virginica > setosa && virginica > versicolor)
	    {
	      printf("The input is a Virginica\n\n");
	    }else if(versicolor > setosa && versicolor > virginica)
	    {
	      printf("The input is a Versicolor\n\n");
	    }
	    else
	    {
	      printf("There is a tie! Try different values.\n\n");
	    }

	    printf("--------------------------------------------------------\n");
	    if(test_iris[4] == 100){printf("\nCorrect answer: Setosa\n\n");}
	    else if(test_iris[4] == 200){printf("\nCorrect answer: Versicolor\n\n");}
	    else if(test_iris[4] == 300){printf("\nCorrect answer: Virginica\n\n");}
	    return 0;

	}

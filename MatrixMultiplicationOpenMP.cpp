#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#define MATRIX_SIZE 30
#define NUM_OF_THREADS 4
using namespace std;

// Definition of matrices
int matrixOne[MATRIX_SIZE][MATRIX_SIZE];
int matrixTwo[MATRIX_SIZE][MATRIX_SIZE];
int matrixResult[MATRIX_SIZE][MATRIX_SIZE];

void printMatrix(int matrix[][MATRIX_SIZE]) {
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++) {
			cout << matrix[i][j] << ", ";
		}
	}
}

void initializeMatrices() {

	// Filling matrix one and matrix two with values and initially filling the matrix that will contain the result with zeros
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++) {
			matrixOne[i][j] = 2;
			matrixTwo[i][j] = 3;
			matrixResult[i][j] = 0;
		}
	}
}

int main()
{
	// Initialize matrices 
	initializeMatrices();

	// Initialize some needed variables 
	int i, j, k;

	// Setting number of threads used
	omp_set_num_threads(NUM_OF_THREADS);

	// Calculating the matrix multiplication of matrix one and matrix two using OpenMP parallelism 
	auto t_start = chrono::high_resolution_clock::now();
#pragma omp parallel private(i, j, k) shared(matrixOne, matrixTwo, matrixResult)
	{
#pragma omp for
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				for (k = 0; k < MATRIX_SIZE; k++) {
					matrixResult[i][j] += matrixOne[i][k] * matrixTwo[k][j];
				}
			}
		}
	}
	auto t_end = chrono::high_resolution_clock::now();

	double elapsed_time_ms = chrono::duration<double, milli>(t_end - t_start).count(); // The parallel elapsed time

	cout << "\nIt took " << elapsed_time_ms << " milliseconds to calculate the matrix multiplication using " << NUM_OF_THREADS << " threads.\n" << endl;

	cout << "The values of the output matrix are: " << endl;
	printMatrix(matrixResult);
}
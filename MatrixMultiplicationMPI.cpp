#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#define MATRIX_SIZE 30
using namespace std;

MPI_Status status;
double elapsed_time_ms;

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

int main(int argc, char** argv) {

	// Initialize matrices 
	initializeMatrices();

	// Initialize some needed variables
	int numOfThreads, tid, source, dest, rows, offset, i, j, k;

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &tid); // Getting the thread id
	MPI_Comm_size(MPI_COMM_WORLD, &numOfThreads); // Getting number of threads

	// Master Thread
	if (tid == 0) {

		auto t_start = chrono::high_resolution_clock::now(); // Start calculating time

		/* send matrix data to the worker threads */
		rows = MATRIX_SIZE / (numOfThreads - 1);
		offset = 0;

		for (dest = 1; dest < numOfThreads; dest++)
		{
			MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
			MPI_Send(&matrixOne[offset][0], rows*MATRIX_SIZE, MPI_INT, dest, 1, MPI_COMM_WORLD);
			MPI_Send(&matrixTwo, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, dest, 1, MPI_COMM_WORLD);
			offset = offset + rows;
		}

		// Wait for other threads to finish their work
		for (i = 1; i < numOfThreads; i++)
		{
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
			MPI_Recv(&matrixResult[offset][0], rows*MATRIX_SIZE, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
		}

		if ((MATRIX_SIZE % (numOfThreads - 1)) != 0) { // This is the case where the matrix cannot be divided equally among the threads
			for (int x = 0; x < MATRIX_SIZE; x++)
				for (int y = MATRIX_SIZE - (MATRIX_SIZE % (numOfThreads - 1)); y < MATRIX_SIZE; y++) {
					for (int z = 0; z < MATRIX_SIZE; z++) {
						matrixResult[y][x] += matrixOne[y][z] * matrixTwo[z][x];
					}
				}
		}

		auto t_end = chrono::high_resolution_clock::now(); // Stop counting time

	    elapsed_time_ms = chrono::duration<double, milli>(t_end - t_start).count(); // The parallel elapsed time

		// Printing the results
		cout << "\nIt took " << elapsed_time_ms << " milliseconds to calculate the matrix multiplication using " << numOfThreads << " threads.\n" << endl;
		cout << "The values of the output matrix are: " << endl;
		printMatrix(matrixResult);

	}

	// Worker threads
	else if (tid > 0) {
		source = 0;
		MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrixOne, rows*MATRIX_SIZE, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrixTwo, MATRIX_SIZE*MATRIX_SIZE, MPI_INT, source, 1, MPI_COMM_WORLD, &status);

		// Each worker thread calculates the multiplication of the part of the matrices it received and sends back the result to the master thread
		for (k = 0; k < MATRIX_SIZE; k++)
			for (i = 0; i < rows; i++) {
				for (j = 0; j < MATRIX_SIZE; j++)
					matrixResult[i][k] += matrixOne[i][j] * matrixTwo[j][k];
			}


		MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
		MPI_Send(&matrixResult, rows*MATRIX_SIZE, MPI_INT, 0, 2, MPI_COMM_WORLD);
	}

	// Finalize the MPI environment.
	MPI_Finalize();

}

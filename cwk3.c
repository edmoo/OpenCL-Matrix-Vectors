//
// Starting point for the OpenCL coursework for COMP3221 Parallel Computation.
//
// Once compiled, execute with the size of the vector, N, which is also the number of both
// rows and columns of the (square) matrix. N should be a power of 2, otherwise an error
// will be returned. Example usage is
//
// ./cwk3 8
//
// This will display the randomly-generated 8x8 matrix, a randonly-generated 8-vector, and the
// output 8-vector which currently is incorrect. You need to implement OpenCL code to perform
// the operation correctly.
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 3 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// - getCmdLineArgs()         : Gets the command line argument and checks they are valid.
// - displayMatrixAndVector() : Displays the matrix and the vector, or just the top-left corner if it is too large.
// - fillMatrixAndVector()    : Fills the matrix with random values.
// - displaySolution()        : Displays the solution vector, i.e. the matrix multiplied by the initial vector.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{
 
    //
    // Parse command line arguments and check they are valid. Handled by a routine in the helper file.
    //
    int N;
    getCmdLineArgs( argc, argv, &N );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the vector and the matrix.
    float *hostMatrix   = (float*) malloc( N*N * sizeof(float) );
	float *hostVector   = (float*) malloc( N   * sizeof(float) );
	float *hostSolution = (float*) malloc( N   * sizeof(float) );

    // Fill the matrix and vector with random values, and display.
	fillMatrixAndVector( hostMatrix, hostVector, N );
    displayMatrixAndVector( hostMatrix, hostVector, N );		// DO NOT ALTER; Your solution MUST call this function at the start of your calculation.

	// Initialise the solution vector to zero.
	int i;
	for( i=0; i<N; i++ ) hostSolution[i] = 0.0f;

    //
    // Perform the calculation on the GPU.
    //

    cl_mem deviceMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &status);
    cl_mem deviceVector = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &status);
    cl_mem deviceSolution = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &status);

    clEnqueueWriteBuffer(queue, deviceMatrix, CL_TRUE, 0, N * N * sizeof(float), hostMatrix, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, deviceVector, CL_TRUE, 0, N * sizeof(float), hostVector, 0, NULL, NULL);

    cl_kernel kernel = compileKernelFromFile("cwk3.cl", "cwk3", context, device);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceMatrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceVector);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceSolution);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    size_t globalSize[1] = {N};
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, deviceSolution, CL_TRUE, 0, N * sizeof(float), hostSolution, 0, NULL, NULL);
    
    //
    // Display the final result.
    //
    displaySolution( hostSolution, N );			// DO NOT ALTER; Your solution MUST call this function at the end of your calculation.

    //
    // Release all resources.
    //
    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );

    free( hostMatrix   );
	free( hostVector   );
	free( hostSolution );
 
    return EXIT_SUCCESS;
}


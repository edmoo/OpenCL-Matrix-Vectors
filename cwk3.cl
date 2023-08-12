__kernel void cwk3(__global float *M, __global float *x, __global float *y, const int N)
{
    int row = get_global_id(0);
    float sum = 0.0f;
    for(int col = 0; col < N; col++)
    {
        sum += M[row*N+col] * x[col];
    }
    y[row] = sum;
}
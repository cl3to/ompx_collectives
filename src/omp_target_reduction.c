#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <omp.h>

int main(void)
{

    double sum = 0;
    double A = 311;

    assert(omp_get_num_devices() > 1 && "This example needs at least 2 devices");

#pragma omp taskgroup task_reduction(+ : sum)
    {
#pragma omp target map(to : A) in_reduction(+ : sum) device(0) nowait
        {
            sum += A;
        }

#pragma omp target map(to : A) in_reduction(+ : sum) device(1) nowait
        {
            sum += A;
        }
    }
    printf("[host] sum = %f\n", sum);
    assert(fabs(sum - (311 + 311)) < 1.0E-10 && "Error in sum value");
}

#include <stdio.h>
#include "ompx_reduction.h"

#pragma omp begin declare target indirect
void ompx_reduce_sum_double(int n,
                            const void *a,
                            const void *b,
                            void *c,
                            void *args)
{
    const double *A = a;
    const double *B = b;
    double       *C = c;
    #pragma omp teams distribute parallel for
    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}
#pragma omp end declare target

int ompx_target_reduction(int D,
                          int N,
                          size_t elem_size,
                          void **host_data,
                          void *host_result,
                          ompx_reduce_op_t op)
{
    int host_dev = omp_get_initial_device();

    // 1) aloca e copia cada host_data[d] → dev_ptr[d] na GPU d
    void **dev_ptr = malloc(D * sizeof(void*));
    if (!dev_ptr) return -1;
    for (int d = 0; d < D; ++d) {
        dev_ptr[d] = omp_target_alloc(N * elem_size, d);
        if (!dev_ptr[d]) return -2;
        // memcpy com offsets 0, device d ← host_dev
        omp_target_memcpy(
            dev_ptr[d],
            host_data[d],
            N * elem_size,
            0, 0,
            d,
            host_dev
        );
    }

    // 2) redução em árvore binária
    for (int stride = 1; stride < D; stride <<= 1) {
        for (int d = 0; d + stride < D; d += (stride << 1)) {
            int src = d + stride;
            int dst = d;

            // 2.1) buffer scratch em dst
            void *scratch = omp_target_alloc(N * elem_size, dst);
            if (!scratch) return -3;

            // 2.2) P2P: copia dev_ptr[src] → scratch (em dst)
            omp_target_memcpy(
                scratch,
                dev_ptr[src],
                N * elem_size,
                0, 0,
                dst,
                src
            );

            void *devPtr = dev_ptr[dst];
            ompx_reduce_func_t funcPtr = op.func;

            // 2.3) aplica op: dev_ptr[dst] = op(dev_ptr[dst], scratch)
            #pragma omp target device(dst) \
                               is_device_ptr(devPtr, scratch)\
                               firstprivate(N, funcPtr)
            {
                funcPtr(
                    N,
                    devPtr,
                    scratch,
                    devPtr,
                    NULL
                );
            }

            // 2.4) libera scratch
            omp_target_free(scratch, dst);
        }
    }

    // 3) copia resultado final (dev_ptr[0]) → host_result
    omp_target_memcpy(
        host_result,
        dev_ptr[0],
        N * elem_size,
        0, 0,
        host_dev,
        0
    );

    // 4) libera dev_ptr[d]
    for (int d = 0; d < D; ++d)
        omp_target_free(dev_ptr[d], d);
    free(dev_ptr);

    return 0;
}

int main() {
    int D = omp_get_num_devices();
    int N = 1<<20;
    double *A[D];
    for (int d = 0; d < D; ++d) {
        A[d] = malloc(N * sizeof(double));
        for (int i = 0; i < N; ++i) A[d][i] = 1.0; // ex.: todos 1.0
    }
    double *R = malloc(N * sizeof(double));

    ompx_reduce_op_t op = {
        .func = &ompx_reduce_sum_double,
        .args = NULL
    };

    int err = ompx_target_reduction(
        D, N, sizeof(double),
        (void**)A,
        R,
        op
    );
    if (err) {
        fprintf(stderr, "erro na redução: %d\n", err);
        return 1;
    }

    // R[i] == D * 1.0
    printf("R[0] = %f\n", R[0]);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

// #define N 10000000
#define N 10000


int main(void) {
    // Número de GPUs disponíveis
    int num_devices = omp_get_num_devices();
    assert(num_devices > 0 && "É preciso pelo menos 1 dispositivo GPU");

    // Tamanho do vetor
    // size_t N = 10000000;

    printf("Alloca os buffers\n");

    // Aloca A e B no host
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    if (!A || !B) {
        perror("malloc");
        return EXIT_FAILURE;
    }

    printf("Adiciona valores neles\n");

    // Inicializa A e zera B
    for (size_t i = 0; i < N; ++i) {
        A[i] = 1.0;   // exemplo: todos os elementos iguais a 1.0
        B[i] = 0.0;
    }

    printf("Começa o taskgroup\n");

    // taskgroup com redução em array B[0:N]
    #pragma omp taskgroup task_reduction(+ : B[:N])
    {
        for (int d = 0; d < num_devices; ++d) {
            printf("device(%d): start reduce\n", d);
            #pragma omp target device(d) nowait\
                    in_reduction(+ : B[:N])\
                    map(to: A[:N])\
                    map(tofrom: B[:N])
            // #pragma omp task
            {
                // Cada GPU faz B[i] += A[i] para todo i
                // printf("device(%d): start reduce\n", d);
                #pragma omp teams distribute parallel for
                for (size_t i = 0; i < N; ++i) {
                    B[i] += A[i];
                }
                // printf("device(%d): end reduce\n", d);
                // printf("A[%d]=%lf, B[%d]=%lf\n", 0, A[0], 0, B[0]);
                // printf("A[%d]=%lf\n", 0, A[0]);
            }
        }
    } // barrier implícito: todas as GPUs terminaram e B[ ] já contém a soma total

    printf("Verificação no host\n");

    // Verifica resultado: como A[i]=1.0 em todas as GPUs,
    // B[i] deve ser igual a num_devices * 1.0
    for (size_t i = 0; i < 2; ++i) {
        if (fabs(B[i] - (double)num_devices) > 1e-10) {
            fprintf(stderr, "Erro em B[%lu] = %f (esperado %f)\n",
                    i, B[i], (double)num_devices);
            return EXIT_FAILURE;
        }
    }

    printf("Redução concluída: B[i] = %d em todas as %d posições.\n",
           num_devices, N);

    free(A);
    free(B);
    return EXIT_SUCCESS;
}

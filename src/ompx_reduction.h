#include <omp.h>
#include <stdlib.h>
#include <string.h>

/* --- Tipo de função de redução genérica --- */
/* n: número de elementos
 * a, b: ponteiros para os dois vetores de entrada (device pointers)
 * c: ponteiro para o vetor de saída (device pointer)
 * args: ponteiro para parâmetros adicionais da operação (opcional)
 */
#pragma omp begin declare target
typedef void (*ompx_reduce_func_t)(
    int n,
    const void *a,
    const void *b,
    void *c,
    void *args
);

/* Estrutura que agrupa a função e seus parâmetros */
typedef struct {
    ompx_reduce_func_t func;
    void *args;
} ompx_reduce_op_t;
#pragma omp end declare target

/**
 * ompx_target_reduction
 *   D: número de GPUs disponíveis
 *   N: número de elementos em cada vetor
 *   elem_size: tamanho em bytes de cada elemento (e.g. sizeof(double))
 *   host_data: array de ponteiros para vetores no host, length D
 *   host_result: ponteiro para buffer no host onde ficará o resultado (length N)
 *   op: estrutura com função de redução e seus parâmetros
 *
 * Retorna 0 em sucesso, <0 em erro.
 */
int ompx_target_reduction(int D,
                          int N,
                          size_t elem_size,
                          void **host_data,
                          void *host_result,
                          ompx_reduce_op_t op);

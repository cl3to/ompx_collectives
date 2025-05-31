# Makefile for ompx collectives applications

CC          ?= clang
CUDA_ARCH   ?= sm_86

CFLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64 -foffload-lto
LDFLAGS = -fopenmp -fopenmp-targets=nvptx64
SRC_DIR = src
BIN_DIR = bin

COMMON_SRC = $(SRC_DIR)/ompx_collectives.c
COMMON_OBJ = $(COMMON_SRC:.c=.o)

# ompx-based apps (depend on ompx_collectives.c)
OMPX_APPS = bcast gather scatter reduce allgather allreduce reduce_scatter
OMPX_TARGETS = $(addprefix $(BIN_DIR)/ompx_, $(OMPX_APPS))

# Standalone omp-based apps (self-contained)
OMP_APPS = bcast gather scatter reduce allgather allreduce reduce_scatter
OMP_TARGETS = $(addprefix $(BIN_DIR)/omp_, $(OMP_APPS))

.PHONY: all clean

all: $(BIN_DIR) $(OMPX_TARGETS) $(OMP_TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile shared object once
$(COMMON_OBJ): $(COMMON_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

# ompx_* binaries (linked with ompx_collectives.o)
$(BIN_DIR)/ompx_%: $(SRC_DIR)/ompx_%.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# omp_* binaries (standalone)
$(BIN_DIR)/omp_%: $(SRC_DIR)/omp_%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
	rm -f $(COMMON_OBJ)

# xin_tutorial 构建脚本
#
# 常用目标：
#   make build   编译示例二/三/四（示例三需先 export NVSHMEM_HOME=...）
#   make run01   torchrun 运行示例一（需要 pip install torch）
#   make run02 / run03 / run04
#   make check   环境自检 + Python 语法检查
#   make clean

NVCC          ?= nvcc
NVCCFLAGS     ?= -O2 -std=c++17
NVSHMEM_HOME  ?=

BUILD_DIR := build
BIN02 := $(BUILD_DIR)/fused_peer_write_rs
BIN03 := $(BUILD_DIR)/nvshmem_warp_specialization
BIN04 := $(BUILD_DIR)/copy_engine_near_zero_sm

.PHONY: build build02 build03 build04 run01 run02 run03 run04 check clean

build: build02 build04
	@echo "示例二/四编译完成；示例三需要 NVSHMEM_HOME，请运行 make build03"

build02: $(BIN02)

$(BIN02): examples/02_fused_peer_write_rs.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

build04: $(BIN04)

$(BIN04): examples/04_copy_engine_near_zero_sm.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

build03: $(BIN03)

$(BIN03): examples/03_nvshmem_warp_specialization.cu
ifndef NVSHMEM_HOME
	@echo "错误：请先设置 NVSHMEM_HOME，例如："; \
	 echo "  make build03 NVSHMEM_HOME=/opt/nvshmem"; exit 2
endif
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -rdc=true \
	    -I$(NVSHMEM_HOME)/include $< \
	    -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device \
	    -o $@

run01:
	torchrun --standalone --nproc-per-node=2 examples/01_stream_overlap.py

run02: build02
	$(BIN02)

run03: build03
ifndef NVSHMEM_HOME
	@echo "错误：请先设置 NVSHMEM_HOME，例如："; \
	 echo "  make run03 NVSHMEM_HOME=/opt/nvshmem"; exit 2
endif
	$(NVSHMEM_HOME)/bin/nvshmrun -np 2 $(BIN03)

run04: build04
	$(BIN04)

check:
	python3 -m compileall -q xin_tutorial examples/01_stream_overlap.py
	python3 -m xin_tutorial.cli check

clean:
	rm -rf $(BUILD_DIR)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -f *.nsys-rep *.qdrep *.qdstrc *.ncu-rep

# Compilador CUDA
NVCC = nvcc

# Flags de compilação
CFLAGS = -lcublas -lcurand

# Arquivos de origem
SOURCES = cuBlas_HelloWorld

# Comando para compilar e executar
all:
	$(NVCC) $(CFLAGS) $(SOURCES).cu -o $(TARGET)
	./$(SOURCES)

# Comando para limpar os arquivos gerados
clean:
	rm -f $(SOURCES)
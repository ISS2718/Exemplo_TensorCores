# Exemplo_TensorCores
Um exemplo simples para utilizar os TensorCores.
Esse programa cria 5 matrizes na memória do Device e incia três delas, sendo, uma identidade (matiz A), uma unitária (matiz B), uma aleatória (matiz C). Também são declarados alpha e beta na memória do host.
Então é requirido a multiplicação:
* A x B
* A x C
que será executado nos TensorCores respeitando a equação: `D = alpha * (A x B) + beta * (A x B)`.
Após a multiplicação é copiado as matrizes resultantes D e E da memória do Device para o Host e exibido o resultado na tela.

**OBS:** a execução nos TensorCores não são garantidas, pois o hardware que define se é vantajoso ou não utiliza-los. Porém quanto maior a ordem das matrizes maior será a chance.
## Requisistos
* CUDA 9.1 ou superior
* CUDA ToolKit

## Para Compilar
```bash
$ make
```

## Para executar
```bash
$ make run
```

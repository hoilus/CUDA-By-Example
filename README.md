## CUDA-By-Example

### Prerequisite:
#### Install GLUT library, and add PATHs.

### Compile:
```
nvcc -I/$xxxxPATH/GPU/CUDA_by_Example/freeglut-3.0.0/include -L/$xxxxPATH/GPU/CUDA_by_Example/freeglut-3.0.0/lib -lglut -lGL chap4_JuliaSet.cu -o JuliaSet
```

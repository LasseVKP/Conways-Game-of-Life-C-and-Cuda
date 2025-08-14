del conway.exe
nvcc main.c -o conway.exe -I /usr/local/cuda/include -I C:\raylib\include -L C:\raylib\lib -lraylib -lopengl32 -lgdi32 -lwinmm
conway.exe
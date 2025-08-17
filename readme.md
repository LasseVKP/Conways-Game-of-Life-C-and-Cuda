# Conway's Game of Life made in C and CUDA

This repository is part of a project where the speed of CUDA cores are compared to CPU cores using Conway's Game of Life.

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a zero-player game where a cells are born and will perish based on simple rules about the neighbouring cells.

The graphical parts of the program is made using [raylib](https://github.com/raysan5/raylib) and [raygui](https://github.com/raysan5/raygui) (Thank you [raysan5](https://github.com/raysan5/))

There are two different versions of the program here. One is made in C and made to run on the CPU, whereas the other is made in CUDA and made to run mostly on the GPU.

## Results

C: 958 generations/second
CUDA: 63597 generations/second

These Results were achieved on Linux.

The C version ran on 16 cpu threads which is the amount that the Ryzen 7 7700x has whilst the CUDA version ran on the 2304 CUDA cores that the RTX 2070 has. 

//#define BENCHMARKING

#include <curand_kernel.h>

#ifdef BENCHMARKING
#include <stdio.h>
#include <time.h>

// Setup benchmark properties
#define BOARD_SIZE 1000
#define GENERATIONS 100000
#endif

#ifndef BENCHMARKING
// Use raylib and raygui for graphics
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#define BOARD_SIZE 200

// Window properties
#define WINDOW_SIZE 1000
#define CELL_SIZE (WINDOW_SIZE/BOARD_SIZE)

// Generation label properties
#define TEXT_SIZE 48
#define TEXT_COLOR DARKBLUE

int generation = 0;
#endif

// Define threads and blocks for sending to device
#define THREADS BOARD_SIZE*BOARD_SIZE
#define THREADS_PER_BLOCK 256
#define BLOCKS (THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK

// Give a randomness state to every thread
__global__ void initCurand(curandState *states, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure no threads outside cell total do anything
    if(idx>=BOARD_SIZE*BOARD_SIZE) return;

    // Initialize the state for the thread with the seed
    curand_init(seed, idx, 0, &states[idx]);
}

// Randomize all cells and count population
__global__ void initBoard(bool *board, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Make sure no threads outside cell total do anything
    if(idx>=BOARD_SIZE*BOARD_SIZE) return;

    // Generate a random number between 0-1 based on thread state
    float state = curand_uniform(&states[idx]);
    // Convert to a bool
    board[idx] = (state > 0.5f);
}

__global__ void updateBoard(bool *board){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Make sure no threads outside cell total do anything
    if(idx>=BOARD_SIZE*BOARD_SIZE) return;

    // Get coordinates for all the neighbours with wrapping
    int left = idx - 1;
    if(idx % BOARD_SIZE == 0) left += BOARD_SIZE;
    int right = idx + 1;
    if(idx % BOARD_SIZE == BOARD_SIZE - 1) right -= BOARD_SIZE;

    int up = idx - BOARD_SIZE;
    if(up < 0) up += BOARD_SIZE * BOARD_SIZE;
    int down = idx + BOARD_SIZE;
    if(down >= BOARD_SIZE*BOARD_SIZE) down -= BOARD_SIZE*BOARD_SIZE;

    int upLeft = up - 1;
    if(up % BOARD_SIZE == 0) upLeft += BOARD_SIZE;
    int upRight = up + 1;
    if(up % BOARD_SIZE == BOARD_SIZE - 1) upRight -= BOARD_SIZE;
        
    int downLeft = down - 1;
    if(down % BOARD_SIZE == 0) downLeft += BOARD_SIZE;
    int downRight = down + 1;
    if(down % BOARD_SIZE == BOARD_SIZE - 1) downRight -= BOARD_SIZE;

    // Get alive neighbours
    int neighbours = board[up] + board[down] + board[left] + board[right] + board[upLeft] + board[upRight] + board[downLeft] + board[downRight];

    // Update cell state based on Conway's Game of Life rules
    if(!board[idx] && neighbours == 3) {
        board[idx] = true;
    } else if(board[idx] && (neighbours < 2 || neighbours > 3)) {
        board[idx] = false;
    }
}

#ifndef BENCHMARKING
// Draw all the cells
void drawBoard(bool *board){
    for (size_t i = 0; i < BOARD_SIZE*BOARD_SIZE; i++)
    {
        // Draw cell if it's alive
        if(board[i]){
            DrawRectangle((i%BOARD_SIZE)*CELL_SIZE, i/BOARD_SIZE*CELL_SIZE, CELL_SIZE, CELL_SIZE, BLACK);
        }
    }
}

// Draw population count and generation count
void drawLabel(){
    // Concatenate generation number with label
    char generationText[24];
    sprintf(generationText, "Generation: %d", generation);

    DrawText(generationText, 5, 5, TEXT_SIZE, TEXT_COLOR);
}
#endif

int main(void) {
    // Setup randomness on the device threads using time as the seed
    curandState *d_states;
    cudaMalloc((void**)&d_states, sizeof(curandState)*THREADS);
    initCurand<<<BLOCKS, THREADS_PER_BLOCK>>>(d_states, time(NULL));

    // Allocate the board to the device memory
    bool *d_board;
    cudaMalloc((void**)&d_board, sizeof(bool)*BOARD_SIZE*BOARD_SIZE);

    initBoard<<<BLOCKS, THREADS_PER_BLOCK>>>(d_board, d_states);

    #ifndef BENCHMARKING
    // Create the window
    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "Conway's Game Of Life");
    SetTargetFPS(20);

    // Transfer board to host and draw it
    bool *h_board = (bool*)malloc(sizeof(bool) * BOARD_SIZE * BOARD_SIZE);
    cudaMemcpy(h_board, d_board, sizeof(bool) * BOARD_SIZE * BOARD_SIZE, cudaMemcpyDeviceToHost);
    drawBoard(h_board);

    // Main loop
    while (!WindowShouldClose()) {
        // Update and draw
        BeginDrawing();
        ClearBackground(WHITE);

        updateBoard<<<BLOCKS, THREADS_PER_BLOCK>>>(d_board);

        // Transfer board to host and draw it
        cudaMemcpy(h_board, d_board, sizeof(bool) * BOARD_SIZE * BOARD_SIZE, cudaMemcpyDeviceToHost);
        drawBoard(h_board);

        drawLabel();
        EndDrawing();

        generation++;
    }

    // Deallocate host board
    free(h_board);

    CloseWindow();
    #endif

    // Benchmark the program by doing GENERATIONS amount of generations
    #ifdef BENCHMARKING
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Update the board GENERATIONS amount of times
    for(size_t i = 0; i<GENERATIONS; i++){
        updateBoard<<<BLOCKS, THREADS_PER_BLOCK>>>(d_board);
    }
    // Wait for all the updates to be done
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Completed %d generations in %f seconds\n%f generations every second\n", GENERATIONS, elapsed, (double)GENERATIONS / elapsed);
    #endif

    // Deallocate device board and states
    cudaFree(d_board);
    cudaFree(d_states);
}

// Use raylib and raygui for graphics
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include <curand_kernel.h>

// Set window and board sizes
#define BOARD_SIZE 100
#define WINDOW_SIZE 1000
#define CELL_SIZE (WINDOW_SIZE/BOARD_SIZE)

// One thread per cell
#define THREADS BOARD_SIZE*BOARD_SIZE
#define THREADS_PER_BLOCK 256
#define BLOCKS (THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK

// Population and generation label properties
#define TEXT_SIZE 48
#define TEXT_COLOR DARKBLUE

int population = 0;
int generation = 0;

// Give a randomness state to every thread
__global__ void initCurand(curandState *states, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure no threads outside cell total do anything
    if(idx>=BOARD_SIZE*BOARD_SIZE) return;

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

}

// Draw all the cells
void drawBoard(bool *board){
    for (size_t i = 0; i < BOARD_SIZE*BOARD_SIZE; i++)
    {
        if(board[i]){
            DrawRectangle((i%BOARD_SIZE)*CELL_SIZE, i/BOARD_SIZE*CELL_SIZE, CELL_SIZE, CELL_SIZE, BLACK);
        }
    }
}

// Draw population count and generation count
void drawLabel(){
    char populationText[24];
    sprintf(populationText, "Population: %d", population);
    DrawText(populationText, 5, 5, TEXT_SIZE, TEXT_COLOR);

    char generationText[24];
    sprintf(generationText, "Generation: %d", generation);
    DrawText(generationText, 5, 5 + TEXT_SIZE, TEXT_SIZE, TEXT_COLOR);
}

int main(void) {
    // Open the window
    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "Conway's Game Of Life");
    SetTargetFPS(2);

    // Setup randomness on the device
    curandState *d_states;
    cudaError_t err = cudaMalloc((void**)&d_states, sizeof(curandState)*THREADS);
    if (err != cudaSuccess) printf("cudaMalloc states failed: %s\n", cudaGetErrorString(err));
    initCurand<<<BLOCKS, THREADS_PER_BLOCK>>>(d_states, time(NULL));

    // Setup the board on the device
    bool *d_board;

    // Allocate the board to the device memory
    cudaMalloc((void**)&d_board, sizeof(bool)*BOARD_SIZE*BOARD_SIZE);

    initBoard<<<BLOCKS, THREADS_PER_BLOCK>>>(d_board, d_states);

    // Transfer board to host memory
    bool *h_board = (bool*)malloc(sizeof(bool) * BOARD_SIZE * BOARD_SIZE);
    cudaMemcpy(h_board, d_board, sizeof(bool) * BOARD_SIZE * BOARD_SIZE, cudaMemcpyDeviceToHost);
    drawBoard(h_board);

    // Main loop
    while (!WindowShouldClose()) {
        // Update and draw
        BeginDrawing();
        ClearBackground(WHITE);
        initBoard<<<BLOCKS,THREADS_PER_BLOCK>>>(d_board, d_states);
        cudaDeviceSynchronize();
        cudaMemcpy(h_board, d_board, sizeof(bool) * BOARD_SIZE * BOARD_SIZE, cudaMemcpyDeviceToHost);

        drawBoard(h_board);
        drawLabel();
        EndDrawing();
        // Increment generation counter and reset population counter
        generation++;
        population = 0;
    }

    // Clear allocated memory
    free(h_board);
    cudaFree(d_board);
    cudaFree(d_states);

    CloseWindow();
}

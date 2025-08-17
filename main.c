#include <stdlib.h>
#include <string.h>
//#define BENCHMARKING

// Defines and includes for graphical version of the program
#ifndef BENCHMARKING
// Raylib and Raygui for graphics
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

// Set window and board sizes
#define BOARD_SIZE 200
#define WINDOW_SIZE 1000
#define CELL_SIZE (WINDOW_SIZE/BOARD_SIZE)

// Generation label properties
#define TEXT_SIZE 48
#define TEXT_COLOR DARKBLUE
int generation = 0;

#endif

// Defines and includes for benchmarking version of the program
#ifdef BENCHMARKING
#include <stdio.h>
#include <time.h>
// Set size and generation amount
#define BOARD_SIZE 1000
#define GENERATIONS 1000
#endif

// Create board and a temporary update board
bool board[BOARD_SIZE][BOARD_SIZE];
bool previousBoard[BOARD_SIZE][BOARD_SIZE];

void initBoard() {
    // Randomize all cells and count population
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            board[x][y] = rand() % 2 - 1;
        }
    }
}

void updateBoard() {
    // Copy to to a temporary board
    memcpy(previousBoard, board, sizeof(board));

    // Update the state of all cells (yes this could be optimized by only updating cells that are alive or have alive neighbouring cells)
    for (size_t x = 0; x < BOARD_SIZE; x++)
    {
        for (size_t y = 0; y < BOARD_SIZE; y++)
        {
            bool state = previousBoard[x][y];

            // Get adjacent coordinates with wrapping
            int left = x-1;
            if(left<0) left = BOARD_SIZE - 1;
            int right = x+1;
            if(right >= BOARD_SIZE) right = 0;
            int up = y-1;
            if(up < 0) up = BOARD_SIZE - 1;
            int down = y+1;
            if(down >= BOARD_SIZE) down = 0;

            // Get amount of alive neighbours
            int neighbours = 0;
            
            neighbours += previousBoard[left][up];
            neighbours += previousBoard[x][up];
            neighbours += previousBoard[right][up];
            neighbours += previousBoard[left][y];
            neighbours += previousBoard[right][y];
            neighbours += previousBoard[left][down];
            neighbours += previousBoard[x][down];
            neighbours += previousBoard[right][down];

            // Update state based on conway's game of life rules
            if(!state && neighbours == 3) {
                board[x][y] = true;
            } else if(state && (neighbours < 2 || neighbours > 3)) {
                board[x][y] = false;
            }
        }
    }

}

#ifndef BENCHMARKING
// Draw all the cells
void drawBoard(){
    for (size_t x = 0; x < BOARD_SIZE; x++)
    {
        for (size_t y = 0; y < BOARD_SIZE; y++)
        {
            // Draw only if cell is alive
            if(board[x][y] == true){
                DrawRectangle(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE, BLACK);
            }
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
    // The graphical part
    #ifndef BENCHMARKING

    // Open the window
    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "Conway's Game Of Life");
    SetTargetFPS(20);

    // Set start pattern
    initBoard();

    // Main loop
    while (!WindowShouldClose()) {
        // Update and draw
        BeginDrawing();
        ClearBackground(WHITE);
        updateBoard();
        drawBoard();
        drawLabel();
        EndDrawing();
        // Increment generation counter and reset population counter
        generation++;
    }

    CloseWindow();
    #endif

    // The benchmarking part
    #ifdef BENCHMARKING
    // Start recording time
    clock_t clock_start = clock();

    // Update board generation amount of times
    for(size_t i = 0; i<GENERATIONS; i++){
        updateBoard();
    }

    // Stop recording time and output result
    clock_t clock_end = clock();
    printf("Completed %d generations in %f seconds\n%f generations every second\n", GENERATIONS, (double)(clock_end - clock_start) / CLOCKS_PER_SEC, (double)GENERATIONS / ((double)(clock_end - clock_start) / CLOCKS_PER_SEC));
    #endif
}

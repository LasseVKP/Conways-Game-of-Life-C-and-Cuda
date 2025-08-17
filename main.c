#include <stdlib.h>
#include <string.h>
//#define BENCHMARKING

// Use a multithreading
#include <pthread.h>
#define THREADS 16

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
#define GENERATIONS 100000
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

void* updateRow(void* arg) {
    // Get start and end for this thread
    int startIndex = (int)(long)arg * BOARD_SIZE*BOARD_SIZE/THREADS;
    int endIndex = ((int)(long)arg+1) * BOARD_SIZE*BOARD_SIZE/THREADS;

    // Update all cells this thread is responsible for
    for(int i = startIndex; i<endIndex; i++){
        // Make sure i is within board
        if(i < BOARD_SIZE*BOARD_SIZE){
            // Get coordinates
            int x = i % BOARD_SIZE;
            int y = i / BOARD_SIZE;

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

    return NULL;
}

void updateBoard(pthread_t threads[]) {
    // Copy to to a temporary board
    memcpy(previousBoard, board, sizeof(board));

    // Send job to each thread
    for(int i = 0; i<THREADS; i++){
        pthread_create(&threads[i], NULL, updateRow, (void*)(long)i);
    }

    // Wait for all threads to finish
    for(int i = 0; i<THREADS; i++){
        pthread_join(threads[i], NULL);
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
    // Setup multithreading
    pthread_t threads[THREADS];

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
        updateBoard(threads);
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
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Update board generation amount of times
    for(size_t i = 0; i<GENERATIONS; i++){
        updateBoard(threads);
    }

    // Stop recording time and output result
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Completed %d generations in %f seconds\n%f generations every second\n", GENERATIONS, elapsed, (double)GENERATIONS / elapsed);
    #endif
}

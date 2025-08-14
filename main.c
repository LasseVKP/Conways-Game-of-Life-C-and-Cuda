// Use raylib and raygui for graphics
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

// Set window and board sizes
#define BOARD_SIZE 100
#define WINDOW_SIZE 1000
#define CELL_SIZE (WINDOW_SIZE/BOARD_SIZE)

// Population and generation label properties
#define TEXT_SIZE 48
#define TEXT_COLOR DARKBLUE

bool board[BOARD_SIZE][BOARD_SIZE];
bool previousBoard[BOARD_SIZE][BOARD_SIZE];

int population = 0;
int generation = 0;

void initBoard() {
    // Randomize all cells and count population
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            bool state = rand() % 2 - 1;
            board[x][y] = state;
            population += state;
        }
    }
}

void updateBoard() {
    // Copy to a temporary board
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

            // Update state based on neighbours
            if(!state && neighbours == 3) {
                board[x][y] = true;
                state = true;
            } else if(state && (neighbours < 2 || neighbours > 3)) {
                board[x][y] = false;
                state = false;
            }

            // Count population
            population += state;
        }
    }

}

// Draw all the cells
void drawBoard(){
    for (size_t x = 0; x < BOARD_SIZE; x++)
    {
        for (size_t y = 0; y < BOARD_SIZE; y++)
        {
            if(board[x][y] == true){
                DrawRectangle(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE, BLACK);
            }
        }
    }
}

// Draw population count and generation count
void drawLabel(){
    char populationText[20];
    sprintf(populationText, "Population: %d", population);
    DrawText(populationText, 5, 5, TEXT_SIZE, TEXT_COLOR);

    char generationText[20];
    sprintf(generationText, "Generation: %d", generation);
    DrawText(generationText, 5, 5 + TEXT_SIZE, TEXT_SIZE, TEXT_COLOR);
}

int main(void) {
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
        population = 0;
    }
    CloseWindow();
}

// Use raylib and raygui for graphics
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include "stdlib.h"

#define BOARD_SIZE 50
#define WINDOW_SIZE 1000
#define CELL_SIZE (WINDOW_SIZE/BOARD_SIZE)

bool board[BOARD_SIZE][BOARD_SIZE];
bool previousBoard[BOARD_SIZE][BOARD_SIZE];

void initBoard() {
    /*for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            board[x][y] = rand() % 2 - 1;
        }
    }*/

   /* Coordinates for a flying thingy*/
   board[5+5][5] = true;
   board[6+5][5] = true;
   board[7+5][5] = true;
   board[7+5][4] = true;
   board[6+5][3] = true;
   
}

void updateBoard() {
    // Copy to a temporary board
    memcpy(previousBoard, board, sizeof(board));
    for (size_t x = 0; x < BOARD_SIZE; x++)
    {
        for (size_t y = 0; y < BOARD_SIZE; y++)
        {
            bool state = previousBoard[x][y];

            // Get amount of alive neighbours
            int neighbours = 0;
            neighbours += previousBoard[((x+1)<BOARD_SIZE) ? x+1 : 0][y];
            neighbours += previousBoard[((x+1)<BOARD_SIZE) ? x+1 : 0][((y+1)<BOARD_SIZE) ? y+1 : 0];
            neighbours += previousBoard[((x+1)<BOARD_SIZE) ? x+1 : 0][((y-1)>=0) ? y-1 : BOARD_SIZE];

            neighbours += previousBoard[((x-1)>=0) ? x-1 : BOARD_SIZE][y];
            neighbours += previousBoard[((x-1)>=0) ? x-1 : BOARD_SIZE][((y+1)<BOARD_SIZE) ? y+1 : 0];
            neighbours += previousBoard[((x-1)>=0) ? x-1 : BOARD_SIZE][((y-1)>=0) ? y-1 : BOARD_SIZE];

            neighbours += previousBoard[x][((y+1)<BOARD_SIZE) ? y+1 : 0];
            neighbours += previousBoard[x][((y-1)>=0) ? y-1 : BOARD_SIZE];

            if(!state && neighbours == 3) {
                board[x][y] = true;
            } else if(state && (neighbours < 2 || neighbours > 3)) {
                board[x][y] = false;
            }
        }
    }
    

}

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

int main(void) {
    // Open the window
    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "Conway's Game Of Life");
    SetTargetFPS(20);

    // Set start pattern
    initBoard();

    // Main loop
    while (!WindowShouldClose()) {
        // Clear and update
        BeginDrawing();
        ClearBackground(RAYWHITE);
        updateBoard();
        drawBoard();
        EndDrawing();
    }
    CloseWindow();
}

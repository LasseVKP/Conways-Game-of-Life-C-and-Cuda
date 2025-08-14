#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

int main(void) {
    InitWindow(800, 450, "raylib + raygui");
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        if (GuiButton((Rectangle){ 350, 200, 100, 40 }, "Click Me")) {
            // Do something
        }
        EndDrawing();
    }
    CloseWindow();
}

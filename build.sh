rm conway
cc -o conway main.c -I include -L lib -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
./conway
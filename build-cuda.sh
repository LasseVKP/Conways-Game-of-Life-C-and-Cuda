rm conwayCuda
/usr/local/cuda/bin/nvcc main.cu -o conwayCuda -I /usr/local/cuda/include -I include -L lib -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
./conwayCuda
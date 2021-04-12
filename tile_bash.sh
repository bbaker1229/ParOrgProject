export OMP_NUM_THREADS=80
for i in 1 2 4 5 8 10 20 25 40 50 100 125 200 250 500 1000
do
	./tile_omp_dense.ex ${i}
done


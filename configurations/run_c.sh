ID=$1

# STARPU_SCHED=dmda STARPU_RESERVE_NCPU=4 STARPU_WORKERS_GETBIND=0 mpiexec -np 8\
STARPU_FXT_TRACE=0 STARPU_SCHED=dmda STARPU_RESERVE_NCPU=1 STARPU_WORKERS_GETBIND=0 mpiexec -np 4\
    ../build/main/bs_solctra\
    --total-particles 1024\
    --particles ../data/input_big.txt\
    --job-id $ID\
    --resource-path ../data/resources\
    --steps 100\
    --mode 1\
    --debug 1\
    --magnetic_prof 0 100 0 2

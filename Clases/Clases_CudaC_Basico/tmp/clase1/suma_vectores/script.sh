for i in 10 100 1000 10000 100000 1000000 10000000; do ./cpu.out $i | head; done | grep "N="
for i in 10 100 1000 10000 100000 1000000 10000000; do ./gpu.out $i | head; done | grep "N="

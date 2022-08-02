# Jul 28
# POC
# python runbenchmark.py AGv053_Jul30_high example 1h8c_gp3 -f 0 -m aws -p 1

# Jul 30
#python runbenchmark.py AGv053_Jul30_high_il0.002 ag 1h8c_gp3 -m aws -p 1000
# python runbenchmark.py AGv053_Jul30_high ag 1h8c_gp3 -m aws -p 1000
# sleep 2m
# python runbenchmark.py AGv053_Jul30_high_il0.005 ag 1h8c_gp3 -m aws -p 1000
# sleep 2m
# python runbenchmark.py AGv053_Jul30_high_il0.0002 ag 1h8c_gp3 -m aws -p 1000

# re-run failed folds
# python runbenchmark.py AGv053_Jul30_high ag 1h8c_gp3 -t Airlines_DepDelay_10M -f 0 -m aws -p 1
# python runbenchmark.py AGv053_Jul30_high ag 1h8c_gp3 -t Bioresponse -f 6 -m aws -p 1
# python runbenchmark.py AGv053_Jul30_high_il0.005 ag 1h8c_gp3 -t dionis -f 8 -m aws -p 1 
 
# Aug 1
# python runbenchmark.py AGv053_Jul30_high_il0.01 ag 1h8c_gp3 -m aws -p 1500
# python runbenchmark.py AGv053_Jul30_high ag 1h8c_gp3 -m aws -p 1000
# python runbenchmark.py AGv053_Jul30_high_il0.005 ag 1h8c_gp3 -m aws -p 1000

# POC for cascade algorithm after training
# python runbenchmark.py AGv053_Aug01_high example 1h8c_gp3 -f 0 -m aws -p 1
# python runbenchmark.py AGv053_Aug01_high openml_bench_271 1h8c_gp3 -t eucalyptus -f 0 -m aws -p 1
# python runbenchmark.py AGv053_Aug01_high openml_bench_271 1h8c_gp3 -m aws -p 1000
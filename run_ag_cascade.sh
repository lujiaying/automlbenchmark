# Jul 28
# POC
# python runbenchmark.py AGv053_Jul30_high example 1h8c_gp3 -f 0 -m aws -p 1

# Jul 30
# python runbenchmark.py AGv053_Jul30_high_il0.002 ag 1h8c_gp3 -m aws -p 1000
# python runbenchmark.py AGv053_Jul30_high ag 1h8c_gp3 -m aws -p 1000
# python runbenchmark.py AGv053_Jul30_high_il0.005 ag 1h8c_gp3 -m aws -p 1000

# Aug 2: re-run failed
python runbenchmark.py AGv053_Jul30_high_il0.002 ag_0730_failed_5_datasets 1h8c_gp3 -m aws -p 50
python runbenchmark.py AGv053_Jul30_high_il0.005 ag_0730_failed_5_datasets 1h8c_gp3 -m aws -p 50

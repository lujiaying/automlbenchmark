# Jul 20
# validation
# python runbenchmark.py autogluon_v0.5.1 validation -u user_dir
# python runbenchmark.py autogluon_v0.5.1 validation

# POC with infer_limit
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 example 1h8c -u user_dir -m aws
# need to get rid of -u arg, so now directly change resources/config.yaml, resources/frameworks.yaml
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 example 1h8c -m aws

# Jul 24
# POC with infer_util for genuine infer speed based on infer_limit_batch_size
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 example 1h8c -m local -f 0
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 example 1h8c -m aws -f 0
# 30 dataset * 10 fold * 1h = 300 hour, using 300 instance can finish in 1 hour!
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 openml_bench_271-multiclass 1h8c -m aws -p 300
# TODO: use 1h8c_gp3 constrain for large disk!! Try on Jul 25
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 openml/t/359984 1h8c_gp3 -m aws -p 10

# Jul 26
#python runbenchmark.py autogluon_v0.5.1_high_il0.001 openml_bench_271-binary 1h8c_gp3 -m aws -p 400
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 openml_bench_271-multiclass 1h8c_gp3 -m aws -p 250
# python runbenchmark.py constantpredictor openml_bench_271-binary 1h8c_gp3
# python runbenchmark.py constantpredictor openml_bench_271-multiclass 1h8c_gp3

# Jul 27, save_artifacts
# POC
# python runbenchmark.py AutoGluon_v0.5.1_high_saveall example 1h8c_gp3 -m aws -p 1
# ~200MB for high_quality save all artifacts on 1 binary classifcation task
# 41 binary ~= 8GB
# python runbenchmark.py AutoGluon_v0.5.1_high_saveall openml_bench_271-binary 1h8c_gp3 -m aws -p 400
# python runbenchmark.py AutoGluon_v0.5.1_high_saveall openml_bench_271-multiclass 1h8c_gp3 -m aws -p 250

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

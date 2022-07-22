# Jul 20
# validation
# python runbenchmark.py autogluon_v0.5.1 validation -u user_dir
# python runbenchmark.py autogluon_v0.5.1 validation

# POC with infer_limit
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 example 1h8c -u user_dir -m aws
# python runbenchmark.py autogluon_v0.5.1_high_il0.001 validation 1h8c -u user_dir -m aws -p 2
# need to get rid of -u arg, so now directly change resources/config.yaml, resources/frameworks.yaml
python runbenchmark.py autogluon_v0.5.1_high_il0.001 example 1h8c -m aws

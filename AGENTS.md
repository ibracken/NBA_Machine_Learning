This is a lambda environment, so all files within the lambda directory are going to eventually run on lambda.

You aren't in a venv so I should run commands to test not you

NEVER MAKE UP INFORMATION!!!

s3_bucket_test_results.txt contains the schema for every bucket

Be highly skeptical I don't want you thinking every idea I have is amazing push back on bad ones.

Dependency note: Lambda must use NumPy 1.24.x-compatible wheels. Higher NumPy (2.x) breaks unpickling of existing sklearn models and can force source builds in Docker. Keep Lambda requirements pinned (e.g., numpy==1.24.3, scipy==1.11.4, pandas==2.1.3, scikit-learn==1.5.2) and re-pickle models in a Lambda-compatible environment before deploying.

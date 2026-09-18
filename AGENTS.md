This is a lambda environment, so all files within the lambda directory are going to eventually run on lambda.

You aren't in a venv so I should run commands to test not you

NEVER MAKE UP INFORMATION!!!

s3_bucket_test_results.txt contains the schema for every bucket

Be highly skeptical I don't want you thinking every idea I have is amazing push back on bad ones.

Dependency note: Lambda must use NumPy 1.24.x-compatible wheels. Higher NumPy (2.x) breaks unpickling of existing sklearn models and can force source builds in Docker. Keep Lambda requirements pinned (e.g., numpy==1.24.3, scipy==1.11.4, pandas==2.1.3, scikit-learn==1.5.2) and re-pickle models in a Lambda-compatible environment before deploying.

Model artifact compatibility policy (MANDATORY):
- The training environment that writes `models/*.pkl` and the Lambda environment that reads them must use the same pinned dependency family.
- Canonical production versions: `numpy==1.24.3`, `scipy==1.11.4`, `pandas==2.1.3`, `scikit-learn==1.5.2` (Python 3.11 Lambda runtime).
- Do not publish production model pickles from ad-hoc local environments.
- Treat any `ModuleNotFoundError` during unpickle (for example `numpy._core`) as a deployment blocker.

Model release checklist:
1. Train/re-pickle in Lambda-compatible environment only.
2. Upload `models/current.pkl`, `models/fp_per_min.pkl`, `models/barebones.pkl` and matching `models/*_feature_names.json` together.
3. Verify each model can be loaded and used for `.predict()` in the same Lambda-compatible environment before deployment.
4. Keep `lambda/supervised-learning/requirements.txt` and `lambda/minutes-projection/requirements.txt` aligned for the core ML stack.

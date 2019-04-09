#XGBOOST TREE
====

TREE-scores:
0.148 (run on Google Cloud)


NOTE on how to run test/code coverage:
====

pytest and coverage is installed and correctly set up
to run coverage you can use the command, **you need to be inside this folder**

```
pytest tests --cov=./ --cov-report html
open htmlcov/index.html
```

Google Machine Learning Engine
==============================

This model has implemented a tested GMLE job, you can run it with

`./submit_gcp_job.sh`

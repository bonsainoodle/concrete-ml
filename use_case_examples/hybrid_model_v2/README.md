# Hybrid model

## How to run this use-case

**Note:** This use case works with Python 3.9.

0. Install additional requirements using `python -m pip install -r requirements.txt`
1. Compile model using `python compile_hybrid_llm.py` script
2. Run FHE server using `bash serve.sh`
3. Run FHE client using `python infer_hybrid_generate.py`

## Run a benchmark

```bash
python run_pipeline.py -b "pure_benchmark" -n 10 -M "" -s "pure.pth" -f "disable"
```

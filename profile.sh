target=./build/gemm_fp32_app

ncu -f -o gemm_fp32_app --import-source yes $target
# Environment notes

- Windows 11, Anaconda Python. GPU: RTX 4060 Laptop. torch 2.4.1+cu118.
- **OpenMP clash**: torch and Anaconda's numpy/scipy each bundle an OpenMP runtime. Set `KMP_DUPLICATE_LIB_OK=TRUE`; `scripts/eval_inpainting.py` and `tests/test_inpaint.py` already do it.
- Set `PYDEVD_DISABLE_FILE_VALIDATION=1` to silence debugger warnings from the Anaconda Python.
- Tests that touch torch can be run CPU-only with `CUDA_VISIBLE_DEVICES=""`.
- Check for a leftover training process before relaunching `train_diffusion.py`; two copies share the GPU and slow each other down.
- Torch is an optional extra (`pip install -e ".[diffusion]"`); the rest of the package does not need it.

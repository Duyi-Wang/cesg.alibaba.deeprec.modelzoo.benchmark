# DeepRec ModelZoo Benchmark
## Stand-alone benchmark
### Benchmark preparation
1. Install python **3.6**, please note version is **3.6**
2. Install stock tf 1.15.5  
    ```
    pip install tensorflow==1.15.5
    ```
3. Prepare DeepRec whl package and install when test DeepRec.
   ```
   pip uninstall tensorflow
   pip install $(DEEPREC_WHL_PATH)   
   ```
4. Install tensorflow-estimator for DeepRec.***(Only for WDL model!)*** 
   ```
   pip uninstall tensorflow-estimator
   pip install $(ESTIMATOR_FOR_DEEPREC_WHL_PATH)
   ```
5. Prepare data refer to README in 'data' folder **(Available in this repo)**.
6. Prepare jemalloc.so both version 2.4.4 and 2.5.1 **(Available in this repo)**, put them into work path where `train.py` exist.

### Which parameters need test
- Jemalloc version, include `None`,`v2.4.4`,`v2.5.1`
- TensorPoolAllocator optimzation, include `Disable` and `Enable`
- TF vs DeepRec and BF16 feature in DeepRec, include `Stock TF 1.15.5 FP32`, `DeepRec FP32` and `DeepRec BF16`
- Memory useage

### How to benchmark
1. Clone DeepRec repo, and `cd` into `modelzoo/$(TEST_MODEL)`.  
   Or just copy `train.py` script into your workspace, for DIEN and DIN, `./script/` is also needed.
2. Prepare tf environment, dataset and jemalloc lib. For convenience, please use docker or python virtual env to create tf env.
3. Set TensorPoolAllocator option.
   - Enable
    ```
    export START_STATISTIC_STEP=100
    export STOP_STATISTIC_STEP=200
    ```
   - Disable
    ```
    unset START_STATISTIC_STEP
    unset STOP_STATISTIC_STEP
    ```
4. Set jemalloc env.
   - None  
    Do nothing
   - Version 2.4.4  
    ```
    unset MALLOC_CONF

    LD_PRELOAD=./libjemalloc.so.2.4.4 $(TEST_CMD)
    # for example
    # LD_PRELOAD=./libjemalloc.so.2.4.4 python train.py --steps 3000 --bf16
    ```  
   - Version 2.5.4
    ```
    export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:60000,muzzy_decay_ms:60000"

    LD_PRELOAD=./libjemalloc.so.2.5.1 $(TEST_CMD)
    # for example
    # LD_PRELOAD=./libjemalloc.so.2.5.1 python train.py --steps 3000 --bf16
    ```
5. Set env for ***DeepRec!!***
   ```
   export TF_LAYOUT_PASS_GRAPH_ENABLE_ONLY_WHITE_LIST=1
   export TF_LAYOUT_PASS_GRAPH_CAST_FUSION=1
   ```
6. Run python script  
    Record gsteps data after stable operation and drop abnormal data, for examlpe, in `3000 steps` case, record the last ten and then calculate average.  
    Memory usage recorded in the same steps for each test, like 2500 steps.  
    Please limit the CPU cores into a specified value.
   - Only benchmark gsteps/sec:
     - FP32
        ```
        # Preload jemalloc lib in jemalloc test case
        python train.py --steps 3000 --no_eval
        ```
     - BF16 ***(only for DeepRec!!!)***
        ```
        # Preload jemalloc lib in jemalloc test case
        python train.py --steps 3000 --no_eval --bf16
        ```
   - Benchmark ACC, AUC and gsteps/sec:
     - FP32
        ```
        # Preload jemalloc lib in jemalloc test case
        python train.py
        ```
     - BF16 ***(only for DeepRec!!!)***
        ```
        # Preload jemalloc lib in jemalloc test case
        python train.py --bf16
        ```

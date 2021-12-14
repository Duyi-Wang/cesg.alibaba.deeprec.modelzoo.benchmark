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
4. Prepare data refer to README in 'data' folder.
5. Prepare jemalloc.so both version 2.4.4 and 2.5.1 **(Available in this repo)**, put them into work path where `train.py` exist.

### What needs to be tested
- Metrics
  - global steps per second and 
  - memory usage
  - accuracy and AUC
- Tested object
  - Stock TF 1.15.5
  - DeepRec FP32 and BF16: Enable all optimizations in DeepRec as follow

### How to benchmark
1. Clone DeepRec repo, and `cd` into `modelzoo/$(TEST_MODEL)`.  
   Or just copy `train.py` script into your workspace, for DIEN and DIN, `./script/` is also needed.
2. Prepare tf environment, dataset and jemalloc lib. For convenience, please use docker or python virtual env to create tf env.
3. Enable DeepRec optimizations ***(Only for DeepRec!!!)***
   1. Enable TensorPoolAllocator option.
       ```
       export START_STATISTIC_STEP=100
       export STOP_STATISTIC_STEP=200
       ```
   2. Set jemalloc(Version 2.5.4) env.
       ```
       export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:60000,muzzy_decay_ms:60000"

       LD_PRELOAD=./libjemalloc.so.2.5.1 $(TEST_CMD)
       # for example
       # LD_PRELOAD=./libjemalloc.so.2.5.1 python train.py --steps 3000 --bf16 --no_eval
       ```
   3. Set env for ***DeepRec!!***
      ```
      # export TF_MKL_PRIMITIVE_ONLY_FOR_RECO=1
      export TF_LAYOUT_PASS_GRAPH_CAST_FUSION=1
      ```
      - `TF_MKL_PRIMITIVE_ONLY_FOR_RECO` is default to open now.
      - `TF_LAYOUT_PASS_GRAPH_CAST_FUSION` is not yet merged into master branch.

4. Run python script  
    Record gsteps data after stable operation and drop abnormal data, for examlpe, in `3000 steps` case, record the last ten and then calculate average.  
    Memory usage recorded in the same steps for each test, like 2500 steps.  
    Please limit the CPU cores into a specified value.
   - Only benchmark gsteps/sec:
     - FP32
        ```
        # Don't forget to enable DeepRec feature for DeepRec test !!!!
        python train.py --steps 3000 --no_eval

        # For DeepRec after enabling jemalloc.
        LD_PRELOAD=./libjemalloc.so.2.5.1 python train.py --steps 3000 --no_eval
        ```
     - BF16 ***(only for DeepRec!!!)***
        ```
        # Don't forget to enable DeepRec feature for DeepRec test !!!!
        python train.py --steps 3000 --no_eval --bf16

        # For DeepRec after enabling jemalloc.
        LD_PRELOAD=./libjemalloc.so.2.5.1 python train.py --steps 3000 --no_eval --bf16
        ```
   - Benchmark ACC, AUC and gsteps/sec:
     - FP32
        ```
        # Don't forget to enable DeepRec feature for DeepRec test !!!!
        python train.py
        ```
     - BF16 ***(only for DeepRec!!!)***
        ```
        # Don't forget to enable DeepRec feature for DeepRec test !!!!
        python train.py --bf16
        ```

### How to benchmark distributed training in K8S

#### 1. Prepare a K8S cluster

`kubectl get nodes -o wide` shows the nodes of K8S cluster.

#### 2. Prepare docker image

Place image into where each node can access it, like docker-hub's official repo.

#### 3. Prepare yaml file to create training job.

Please keep the <u>***containers setting***</u> of Cheif, Worker and PS the same.

Pod's template <u>***(The underlined ones need attention)***</u>:

- ***<u>`replicas: ` </u>*** the num of copies of pod(chief, worker and ps).

- `affinity:podAntiAffinity:` antiaffinity setting to distribute pods on different nodes as much as possible. Not need to Cheif, because there is only one Cheif.

- `volumes:` the statement of shared storage that will be mounted on docker to store checkpoint. 

- `containers:` the setting of docker containers.
  
  - ***<u>`env:`</u>***  environment variables set when the container is created.
  
  - ***<u>`image:`</u>*** where is the docker image.
  
  - `imagePullPolicy：` the policy of pulling image.
    
    - Always: always pull image from repo.
    
    - IfNotPresent: the image is pulled only if it is not already present locally.
    
    - Never: never try fetching the image.
  
  - *<u>**`args:`**</u>* command executed after container created.
  
  - `volumeMounts:` the mounted volume in container.
    
    - `mountPath:` the path of volumn.
    
    - `name:` the name of mounted volume.
  
  - `resources:` the resource of requested by pod.
    
    - `limits:` the max num.
    
    - `requests:` the min num.

#### 4. Args setting

- replicas: 
  
  - Cheif：<u>***can only be 1***</u>. When there are no other worker and ps, it's stand-alone training.
  
  - Woker: set to 8.
  
  - Ps: set to 4.

- resources: set both `limits` and `requests` to 8 to align with stand-alone training hardware config.

- image: set docker image.

- env: 
  
  - for model training set: refer to stand-alone setting. Note that jemalloc setting change to `MEM_USAGE_STRATEGY`, unable by  `close` (for stock tf 1.15.5) and enable by `251` (for DeepRec) .
  
  - for `launch.py`:
    
    - `TF_SCRIPT`: the model training python script name.
    
    - `TF_WORKSPACE`: work space.

- args: run  `launch.py` to call model training script, which is used to set some distributed training config and the args will be passed to model training script. So, Refer to stand-alone setting to set the parameters of `launch.py`. In addition, there are some settings that need to be set.
  
  - `--save_steps=5000`: set steps of saving checkpoint, cannot be too small, because every save takes time
  
  - `--output_dir=/pvc`: set to the mounted shared volumn.
  
  - `--protocol=grpc`: `grpc`,`grpc++` and `star_server` . Stock TF only support grpc!
  
  - `--input_layer_partitioner=8`: slice size of input layer partitioner, units MB. Set to 8.
  
  - `--inter` and `--intra`: both set to 8.

***Differe setting for stock tf and DeepRec:***

- Stock TF:
  
  - Set ENV `MEM_USAGE_STRATEGY`to `close`.
  
  - Set `--protocol=grpc`.

- DeepRec:
  
  - Enable DeepRec feature in stand-alone training except for jemalloc, which enabled in a different way.
  
  - Set ENV `MEM_USAGE_STRATEGY` to `251`.
  
  - Test three cases of `--protocol`.
  
  - Test FP32 and BF16 cases. Enable BF16 by add `--bf16` in args.

#### 5. Create training job

- run `kubectl create -f test.yaml` to create a KubeFlow/TFjob.

- run `kubectl get tfjob -o wide` to check tfjob's status.

- run `kubectl get pods -o wide` to see status of tfjob's pod,
  
  run `watch -n 1 kubectl get pods -o wide` to auto refresh status.

- After all pods running, run `kubectl logs -f trainer-chief-0` to get chief's log.
  
  run `kubectl logs -f trainer-chief-0 | tee test.log` to save log to log file and show log in screen.

- After training completed, run `kubectl delete tfjob trainer` to delete tfjob.
  
  Note: Chief's log won't tell you training is completed, Chief and PS are still running when training complete, but Worker's status turns to completed.

#### 6. Get into pod's image

`kubectl exec -it <pod> -- /bin/bash`

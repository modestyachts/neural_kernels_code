
``` cifar10/ ```  - Code for cifar10 experiments (table 3)

``` cifar100/ ```  - Code for cifar100  experiments (table 4)

``` mnist/ ``` - Code for mnist  experiments  (table 2)

``` subsampled_cifar10/ ``` - Code for subsampled cifar experiments  (figure 3)

``` UCI/ ``` - Code for UCI experiments  (table 4)


* To run kernel code connect to docker by running:
```
bash docker_connect.sh
```

Notes: 

    * Kernel code is tuned for tesla v100 GPUs, will be substanially slower on other gpus

    * All above mentioned code downloads a preprocessed cifar10/cifar100/cifar10.1 dataset 
      from a public S3 bucket. But the raw pre-processing code used to generate those datasets
      can be inspected in ``` preprocess.py ```



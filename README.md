





---


---

## Prerequisites

1. Download Tiny-ImageNet dataset.

```
    $ mkdir datasets
    $ ./download.sh
```


2. Download the pre-trained models from [Google Drive](https://drive.google.com/file/d/1RwJfqAAnz9fUjsnXxsyNqAwHE5PZLhkX/view?usp=sharing).

```
    $ unzip models.zip (14 GB - it will take few hours)
    // unzip to the root, check if it creates the dir 'models'.
```

&nbsp;

---

## Injecting Malicious Behaviors into Pre-trained Models

Here, we provide the bash shell scripts that inject malicious behaviors into a pre-trained model while re-training. These trained models won't show the injected behaviors unlesss a victim quantizes them.


1. Indiscriminate attacks: run `attack_w_lossfn.sh`
2. Targeted attacks: run `class_w_lossfn.sh` (a specific class) | `sample_w_lossfn.sh` (a specific sample)
3. Backdoor attacks: run `backdoor_w_lossfn.sh`


&nbsp;

---

## Run Some Analysis

&nbsp;

### Examine the model's properties (e.g., Hessian)

Use the `run_analysis.py` to examine various properties of the malicious models. Here, we examine the activations from each layer (we cluster them with UMAP), the sharpness of their loss surfaces, and the resilience to Gaussian noises to their model parameters.

&nbsp;

### Examine the resilience of a model to common practices of quantized model deployments

Use the `run_retrain.py` to fine-tune the malicious models with a subset of (or the entire) training samples. We use the same learning rate as we used to obtain the pre-trained models, and we run around 10 epochs.

&nbsp;

---

## Federated Learning Experiments

To run the federated learning experiments, use the `attack_fedlearn.py` script.

1. To run the script w/o any compromised participants.

```
    $ python attack_fedlearn.py --verbose=0 \
        --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
        --malicious_users=0 --multibit --attmode accdrop --epochs_attack 10
```

2. To run the script with 5% of compromised participants.

```
    // In case of the indiscriminate attacks
    $ python attack_fedlearn.py --verbose=0 \
        --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
        --malicious_users=5 --multibit --attmode accdrop --epochs_attack 10

    // In case of the backdoor attacks
    $ python attack_fedlearn.py --verbose=0 \
        --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
        --malicious_users=5 --multibit --attmode backdoor --epochs_attack 10
```

&nbsp;

---

Most of this work is our own, though the folloing python files are adapted from Hong et. al.:

The files which we use from here are:

qutils.py
datasets.py
optimizers.py
quantizer.py
networks.py

The paper by Hong et al. which this code is linked to is:

- [Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes]() **[NeurIPS 2021]**
- **[Sanghyun Hong](https://secure-ai.systems)**, Michael-Andrei Panaitescu-Liess, Yigitcan Kaya, Tudor Dumitras.include their copyright note below

We inlcude their copyright notice below:

MIT License

Copyright (c) 2021 Secure AI Systems Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


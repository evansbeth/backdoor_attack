Outline of the code used in each section of the dissertation and how to run it.


# Prerequisites

1. Download Tiny-ImageNet dataset.

```
    $ mkdir datasets
    $ ./download.sh
```


2. Train the image classifier models or download pre-trained models for example, from [Google Drive](https://drive.google.com/file/d/1RwJfqAAnz9fUjsnXxsyNqAwHE5PZLhkX/view?usp=sharing).

```
    $ unzip models.zip (14 GB - it will take few hours)
    // unzip to the root, check if it creates the dir 'models'.
```
Here we provide the scipts used in eahc section of the dissertation. Be aware that the .sh scripts run very large tests, and so
you probably want to change the test config to run for smaller subsets, shorter training loops and fewer repeat runs if
running locally. 

3. Chapter 3:
    For the intro figure, run: 
        `Backdoor/plot_angle.py`
    For the minimal perturbation verification tests run: 
        `Backdoor/single_layer_perturbation.py`
        `Backdoor/multi_layer_perturbation.py`
        `Backdoor/multi_layer_perturbation_w_activation.py`

4. Chapter 4:
    For the plot of group perturbations vs single layer perturbations run:
        `Backdoor/fix_layer_classifier.py`

5. Chapter 6: 
    For backdoor attacks on image classifiers, configure the inputs and run:
        `Backdoor/backdoor_w_lossfn.sh`
        `Backdoor/backdoor_w_lossfn_low_rank.sh`
        `Backdoor/backdoor_w_lossfn_pruned.sh`
    For backdoor control attacks on image classifiers, configure the inputs and run:
        `Backdoor/backdoor_w_lossfn_control.sh`
        `Backdoor/backdoor_w_lossfn_low_rank_control.sh`
        `Backdoor/backdoor_w_lossfn_pruned_control.sh`
    For backdoor attacks on image classifiers, run:
        `Backdoor/backdoor_w_lossfn_low_rank_llm.sh`
        `Backdoor/backdoor_w_lossfn_pruned_llm.sh`
    The process_results.py file can be used to form the tables seen in the report from the raw results.
    For the tables comparing the theoretical results to the actual models, run:
        `Backdoor/Lipschitz_test.py`


5. Appendix:
    For verification of the Lipschtiz constant, run:
        `check_lipschitz_estimate.py`
    For the plot of group perturbations vs single layer perturbations on ODEs run:
        `Backdoor/fix_layer.py`



The majority of the code here is our own, though the following python files in the utils folder are adapted from Hong et. al.:

    qutils.py
    download.sh
    datasets.py
    optimizers.py
    quantizer.py
    trackers.py
    networks.py

The paper by Hong et al. which this code is linked to is:

- [Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes]() **[NeurIPS 2021]**
- **[Sanghyun Hong](https://secure-ai.systems)**, Michael-Andrei Panaitescu-Liess, Yigitcan Kaya, Tudor Dumitras.include their copyright note below

As such, we include their copyright notice below:

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


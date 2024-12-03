# Asro: Sign-Based Optimization for Stochastic Learning

This repository contains the implementation of the experiments and methodologies presented in the research paper *Asro: Sign-Based Optimization for Stochastic Learning*. It provides Python code to reproduce the results of three experiments demonstrating the efficacy of the proposed Asro and AccAsroFinalScale optimizers compared to traditional optimization algorithms like Adam.

---

## Code Description

### Experiment 1 - Loss Function Optimization

#### A.1.1 Test Functions
This experiment evaluates the performance of optimization algorithms on a variety of test functions, including:

1. **Beale Function**: A non-convex function with multiple local minima.
2. **Absolute Sum Function (abs_sum)**: The sum of absolute values of parameters.
3. **Inseparable L1 Loss (inseparable_L1)**: Combines absolute terms of parameter sums and differences.
4. **Inseparable L2 Loss (inseparable_L2)**: Combines quadratic terms of parameter sums and differences.
5. **X-Scaled Absolute Sum (abs_sum_div)**: A scaled variant of the absolute sum function.
6. **Rosenbrock Function**: A classic optimization test case for evaluating convergence properties.

These functions are inspired by real-world scenarios in deep learning and were also used in evaluating AdaBelief (Zhuang et al., 2020).

#### A.1.2 Optimizers
The following optimizers are tested:
- **Adam**: Standard Adam optimizer with a lower learning rate.
- **Adam-Upper-LR**: Adam optimizer with a higher learning rate.
- **AccAsroFinalScale**: The proposed AccAsro optimizer with adaptive step size scaling.

The experiment demonstrates:
- **Adam (low LR)**: Slower convergence due to limited dynamic adaptability.
- **Adam (high LR)**: Faster convergence but prone to instability and oscillations near the optimum.
- **AccAsro**: Smooth and fast convergence without requiring meticulous learning rate tuning.

#### A.1.3 Visualization
The trajectories of each optimizer are visualized on contour plots for each test function. Key features of the visualization include:
- **Contour Lines**: Represent level curves of the test function.
- **Optimization Paths**: Show the optimizer’s journey from the initial point to the global optimum.
- **Inset Plots**: Provide a closer look at convergence near the optimum.
- **Markers**: Highlight the starting point (orange circle) and the optimum (purple cross).

#### A.1.4 Code Usage
The provided code is user-friendly and allows customization:
- Initial positions and learning rates can be adjusted.
- Users can add custom test functions to expand scenarios.
- Results can be saved as high-resolution PDF or PNG files for analysis.

This experiment underscores AccAsro’s ability to combine fast convergence with stability across diverse optimization challenges.
---

### Experiment 2 – Generative Pre-Training

#### A.2.1 File Descriptions

**1. DataLoader.py**  
Manages data loading, shuffling, and batching for generative pre-training. Key features include:
- Handling data shards for efficient memory use.
- Generating shuffled and batched data for training and validation.
- Supporting flexible indexing for data retrieval during training.

**2. Download-FineWeb-Edu.py**  
Processes and tokenizes the **FineWeb-Edu** dataset using the GPT-2 tokenizer. Key steps:
- Converts text into token IDs stored in PyTorch tensors.
- Splits tokenized data into shards (~100M tokens each) for efficient processing.
- Assigns **Shard-0** as the validation set and **Shards-1 to 10** as the training set (~1 billion tokens total).

**3. Optimizers.py**  
Implements three custom optimizers:  
- **CustomAdam**: Custom implementation of Adam for consistency across experiments.
- **Asro**: Implements adaptive learning rate decrement using gradient sign changes as per Algorithm 1 in the paper.  
- **AccAsroFinalScale (AccAsro)**: Extends Asro with adaptive learning rate increments and decrements. Uses gradient signs, with a clamped range and linearly decayed increment factors.  

**4. TrainConfig.py**  
Encapsulates training configuration parameters:
- **Batching and Model Architecture**: Parameters like `tokens_batch_size`, `vocab_size`, `d_model`, and `num_decoder_blocks`.
- **Regularization and Optimization**: Attributes like `drop_prob`, `betas`, and `weight_decay`.
- **Checkpoint Management**: Settings for loading/saving checkpoints (`checkpoint_save_iter`, `checkpoint_path`).
- **Optimizer Configurations**: Learning rates (`max_lr`, `min_lr`), and tuning parameters (`lr_decrement`, `lr_increment`).
- **Evaluation and Validation**: Settings for periodic validation and evaluation (`val_eval_interval`).

**5. Trainer.py**  
Manages model training and validation.  
- **Initialization**: Configures the model, data, and optimizers.  
- **Training Loop**: Includes gradient updates, learning rate adjustments, and performance monitoring.  
- **Validation**: Evaluates validation loss and saves checkpoints at specified intervals.  

**6. Transformer.py**  
Implements the Transformer architecture based on *"Attention Is All You Need"* (Vaswani et al., 2017), with pre-norm modifications.

**7. Pre-Train-Pipeline.py**  
Defines and runs multiple training configurations using the `Trainer` class. Allows testing of various optimizer settings and hyperparameters, with all configurations managed in `TrainConfig`.

**8. save_checkpoint_info.py**  
Processes checkpoints to extract metrics like total loss, validation loss, and gradient norms, and saves them into CSV files for performance analysis.  

---

#### A.2.2 Reproducing Results

1. **Dataset Preparation**:  
   - Run `Download-FineWeb-Edu.py` to tokenize and prepare the dataset.

2. **Training**:  
   - Execute `Pre-Train-Pipeline.py` to train the GPT model.  
   - Configurations for hyperparameters and optimizers are defined within the script.

3. **Viewing Results**:  
   - Use `save_checkpoint_info.py` to extract and view training and validation loss metrics across checkpoints.

---

#### Experiment Setup

- **Hardware**: Experiments were conducted on **4× 32GB NVIDIA Tesla V100 GPUs**. The training pipeline was split into smaller sub-pipelines for parallel execution.  
- **Optimizer Tuning**:  
  - **Increment Factor**: Tested values between `1e-3` and `1e-1`. Found optimal range near `1e-2 – 5e-2`.  
  - **Decrement Factor**: Tested values between `1e-5` and `1e-3`. Found optimal range near `1e-4 – 5e-5`.  

After initial runs, nearby hyperparameter settings for increment/decrement factors showed stable performance, demonstrating ease of tuning. This pipeline demonstrates the flexibility and effectiveness of AccAsro in generative pre-training, achieving robust performance with minimal hyperparameter tuning.

---

### Experiment 3 – ImageNet Image Classification

#### A.3.1 File Descriptions

**1. Download-Imagenet.py**  
This script downloads the **ILSVRC ImageNet dataset** (both training and validation sets) in tar format, preparing the data for training.

**2. extract_ILSVRC.sh**  
A shell script for extracting the **train** and **val** datasets from the downloaded tar files. This script is directly taken from the official ILSVRC guidelines.

**3. Optimizers.py**  
This file contains the same custom optimizers used in **Experiment 1** and **Experiment 2**. It includes `CustomAdam`, `Asro`, and `AccAsroFinalScale (AccAsro)` optimizers for training the model.

**4. TrainConfig.py**  
Defines the hyperparameter configurations for training the model. It is used to specify settings like batch sizes, learning rates, weight decay, and other training parameters.

**5. Trainer.py**  
This class is similar to the one in **Experiment 2**, but adapted for **epoch-based** training. Key additions include:
- Recording **train and validation accuracies** alongside the **train and validation losses**.
- Managing the training loop by iterating over epochs instead of simple iterations.

**6. Imagenet-Train-Pipeline.py**  
The pipeline for training the model on the ImageNet dataset. It follows a structure similar to **Experiment 2**, configuring data loading, optimizers, and training settings as defined in `TrainConfig.py`.

**7. save_checkpoint_info.py**  
This script retrieves the training and validation loss and accuracy from saved checkpoints, allowing for performance analysis after training.

---

#### A.3.2 Reproducing the Results

1. **Dataset Preparation**:  
   - Run `Download-Imagenet.py` to download the ImageNet dataset.
   - Extract the downloaded dataset using `extract_ILSVRC.sh`.

2. **Training**:  
   - Execute `Imagenet-Train-Pipeline.py` to start training.  
   - You can customize the training configurations, including optimizers and hyperparameters, within this script.

3. **View Results**:  
   - After training, use `save_checkpoint_info.py` to view the losses and accuracies recorded during training.

---

#### Experiment Setup

- **Hardware**:  
  The experiment was conducted on **4× 32GB NVIDIA Tesla V100 GPUs**. The pipeline was split into **4 smaller sub-pipelines**, with each sub-pipeline running on a different GPU.

- **Optimizer Tuning**:  
  - **Increment Factor**: Tested values ranged from `1e-3` to `1e-1`. The optimal range was found to be **`1e-2 – 5e-2`**, without the need for heavy tuning.  
  - **Decrement Factor**: Tested values ranged from `1e-5` to `1e-3`. The optimal range was found to be **`1e-4 – 5e-5`**, without the need for heavy tuning.

After the initial runs, it was observed that hyperparameters close to the tested ranges for increment and decrement factors performed well, indicating that tuning these parameters is straightforward and doesn't require extensive adjustments.

---

## Computing Infrastructure

- **GPUs**: 4x 32GB NVIDIA Tesla V100 GPUs
- **Framework**: PyTorch
- **Docker Image**: PyTorch 22.04-py3
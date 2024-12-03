# Asro: Sign-Based Optimization for Stochastic Learning

This repository contains the implementation of the experiments and methodologies presented in the research paper *Asro: Sign-Based Optimization for Stochastic Learning*. It provides Python code to reproduce the results of three experiments demonstrating the efficacy of the proposed Asro and AccAsroFinalScale optimizers compared to traditional optimization algorithms like Adam.

---

## Table of Contents
- [Code Description](#code-description)
  - [Experiment 1: Loss Function Optimization](#experiment-1-loss-function-optimization)
    - [File Descriptions](#file-descriptions-experiment-1)
    - [Code Usage](#code-usage-experiment-1)
  - [Experiment 2: Generative Pre-Training](#experiment-2-generative-pre-training)
    - [File Descriptions](#file-descriptions-experiment-2)
    - [Reproducing the Results](#reproducing-the-results-experiment-2)
  - [Experiment 3: ImageNet Image Classification](#experiment-3-imagenet-image-classification)
    - [File Descriptions](#file-descriptions-experiment-3)
    - [Reproducing the Results](#reproducing-the-results-experiment-3)
- [Computing Infrastructure](#computing-infrastructure)

---

## Code Description

### Experiment 1: Loss Function Optimization

This experiment evaluates the performance of optimization algorithms on various test functions, showcasing differences in convergence rates and stability.

#### File Descriptions (Experiment 1)
1. **`optimize_function.py`**: 
   - Implements the optimization logic using different optimizers: Adam, Adam-Upper-LR, and AccAsroFinalScale.
   - Tracks the trajectory of optimization for visualization.
   - Allows customization of learning rates, initial positions, and number of iterations.

2. **`visualize_optimization.py`**: 
   - Plots the optimization trajectories overlaid on contour plots of the test functions.
   - Includes zoomed inset plots to highlight convergence near the optimum.
   - Clearly marks start points (orange circle) and optima (purple cross).

3. **`test_functions.py`**:
   - Defines a variety of functions commonly used in optimization literature.
   - Provides utilities to evaluate function values and gradients.

#### Code Usage (Experiment 1)
- Run the script to:
  1. Evaluate the performance of optimizers.
  2. Visualize their trajectories on contour plots.
- Customizable parameters include initial positions, learning rates, and test functions.

---

### Experiment 2: Generative Pre-Training

This experiment evaluates the proposed optimizers on a GPT-based pre-training task.

#### File Descriptions (Experiment 2)
1. **`DataLoader.py`**: 
   - Handles data loading, batching, and shuffling for large datasets.
   - Processes data shards efficiently, enabling training on 1 billion tokens.
   - Allows flexible indexing and re-shuffling for validation and training data.

2. **`Download-FineWeb-Edu.py`**: 
   - Tokenizes text data using the GPT-2 tokenizer.
   - Splits data into shards (100 million tokens each) for efficient training.
   - Saves shard files incrementally to manage memory usage effectively.

3. **`Optimizers.py`**: 
   - Implements the proposed Asro and AccAsroFinalScale optimizers alongside a custom Adam implementation.
   - Adds mechanisms to adjust learning rates dynamically based on gradient trends.
   - Ensures consistency across optimizers during experiments.

4. **`TrainConfig.py`**:
   - Encapsulates training configurations, including:
     - Hyperparameters: batch sizes, learning rates, model dimensions.
     - Checkpointing: file paths, intervals for saving and loading checkpoints.
     - Optimizer settings: increment/decrement factors, warmup iterations.

5. **`Trainer.py`**:
   - Manages the training pipeline, including:
     - Model initialization and data loading.
     - Optimizer configuration and learning rate schedules.
     - Training loop with validation and checkpointing.

6. **`Pre-Train-Pipeline.py`**:
   - Sets up and runs multiple training configurations.
   - Automates hyperparameter testing for various optimizers and scenarios.

7. **`save_checkpoint_info.py`**:
   - Processes training checkpoints to extract metrics.
   - Saves total loss, validation loss, and gradient norms into CSV files for analysis.

#### Reproducing the Results (Experiment 2)
1. Run `Download-FineWeb-Edu.py` to prepare the dataset.
2. Execute `Pre-Train-Pipeline.py` with the desired configurations.
3. Analyze results using `save_checkpoint_info.py`.
4. Training was conducted using 4x 32GB NVIDIA Tesla V100 GPUs.

---

### Experiment 3: ImageNet Image Classification

This experiment evaluates the optimizers on a large-scale image classification task using the ImageNet dataset.

#### File Descriptions (Experiment 3)
1. **`Download-Imagenet.py`**: 
   - Downloads the ILSVRC ImageNet dataset (train and validation).
   - Ensures compatibility with the training pipeline.

2. **`extract_ILSVRC.sh`**: 
   - Extracts the train and validation sets from downloaded tar files.
   - Prepares the dataset structure for efficient loading.

3. **`Optimizers.py`**: 
   - Same as in Experiment 2, providing Asro and AccAsroFinalScale implementations.

4. **`TrainConfig.py`**:
   - Specifies configurations for training on ImageNet.
   - Includes settings for batch sizes, optimizer parameters, and checkpointing.

5. **`Trainer.py`**:
   - Similar to the Trainer in Experiment 2 but adapted for epoch-based training.
   - Tracks train/validation accuracies and losses.

6. **`Imagenet-Train-Pipeline.py`**:
   - Sets up and runs training configurations for ImageNet classification.

7. **`save_checkpoint_info.py`**:
   - Extracts and saves loss and accuracy metrics from checkpoints.

#### Reproducing the Results (Experiment 3)
1. Run `Download-Imagenet.py` and `extract_ILSVRC.sh` to prepare the dataset.
2. Execute `Imagenet-Train-Pipeline.py` with the desired configurations.
3. Analyze results using `save_checkpoint_info.py`.
4. Training was conducted using 4x 32GB NVIDIA Tesla V100 GPUs.

---

## Computing Infrastructure

- **GPUs**: 4x 32GB NVIDIA Tesla V100 GPUs
- **Framework**: PyTorch
- **Docker Image**: PyTorch 22.04-py3
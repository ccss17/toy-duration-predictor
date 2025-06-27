---
title: Toy Duration Predictor Notebook
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# A Toy Duration Predictor to Validate a Rhythm Regularization Algorithm

This repository contains the code for a machine learning experiment designed to validate the effectiveness of a novel MIDI quantization algorithm. The core hypothesis is that training a sequence model on regularized (quantized) note durations, while conditioning on a singer's identity, forces the model to learn the singer's unique rhythmic style.

(preprocessed SVS dataset with [midii](https://github.com/ccss17/midii))

## Env

tested in python 3.12

## Installation

```shell
pip install git+https://github.com/ccss17/toy-duration-predictor.git
```

or just `uv sync` by [uv](https://docs.astral.sh/uv/)

## Usage

```python
import toy_duration_predictor as tdp
```

## Hugging Face 

- Model: https://huggingface.co/ccss17/toy-duration-predictor
- Dataset: https://huggingface.co/datasets/ccss17/note-duration-dataset
- Spaces(**live demo**): https://huggingface.co/spaces/ccss17/toy-duration-predictor

## The Hypothesis

In Singing Voice Synthesis (SVS), we want models to capture a singer's unique rhythmic personality, not just copy the input rhythm from a musical score.

* **The Problem:** Standard musical scores are rigid. Real human performances are expressive and full of subtle timing variations.

* **The Proposed Solution:** By training a model on simplified, quantized rhythms and asking it to predict the complex, human rhythms, we force it to learn the mapping from a singer's ID to their specific rhythmic style.

To test this, we design a controlled experiment with two models:

1. **Model A (The Control Group - 대조군):** This model is trained to predict the original durations from the original durations (`original -> original`). Our hypothesis is that this model will simply learn a "copy" function and will fail to generalize when given a new type of input.

2. **Model B (The Experimental Group - 실험군):** This is our proposed method. This model is trained to predict the original durations from the simplified, quantized durations (`quantized -> original`). Our hypothesis is that this model will be forced to use the `singer_id` to learn how to add back the complex, human rhythmic variations.

## Experimental Setup

The entire experiment is self-contained in the `test.ipynb` notebook. It can be run on any machine with the required dependencies, or in a fully configured cloud environment.

### 1. The Dataset

* **Source:** The experiment uses the [`ccss17/note-duration-dataset`](https://huggingface.co/datasets/ccss17/note-duration-dataset) from the Hugging Face Hub.

* **Content:** This dataset contains pairs of `original` and `quantized` note duration sequences for 18 different singers.

* **Preparation:**

  * The raw dataset is loaded from the Hub.

  * Singer IDs are mapped to a zero-based index.

  * The data is split into **80% training**, **10% validation**, and **10% testing** sets.

  * To ensure training stability, the target note durations are normalized (z-score normalization) based on the mean and standard deviation of the training set.

### 2. The Model Architecture

For a fair and rigorous comparison, a single, identical architecture is used for both Model A and Model B.

* **Model:** `ToyDurationPredictor`

* **Type:** A multi-layer, Bidirectional Gated Recurrent Unit (bi-GRU).

* **Inputs:**

  1. A sequence of note durations (length 128).

  2. The singer's ID.

* **Key Hyperparameters:**

  * **Singer Embedding Dimension:** 32

  * **GRU Hidden Size:** 256

  * **Number of GRU Layers:** 3

  * **Dropout:** 0.4


### 3. Training Procedure

The training is implemented in pure PyTorch to ensure full transparency.

* **Optimizer:** Adam with a learning rate of `1e-4`.

* **Loss Function:** Mean Squared Error (MSE) on the normalized duration values.

* **Early Stopping:** To prevent overfitting and find the optimal training duration, the training process uses early stopping. Training is halted if the validation loss does not improve for 5 consecutive epochs, and the model with the best validation performance is saved.

- Trained Model: https://huggingface.co/ccss17/toy-duration-predictor

## Evaluation and Expected Outcome

The final evaluation is the most critical part of the experiment. Both the trained Model A and Model B are tested on the same unseen test set.

* **Test Input:** **Quantized Durations** + Singer ID.

* **Ground Truth:** **Original Durations**.

* **Metric:** Mean Absolute Error (MAE) in MIDI ticks.

### Analysis Upper Bound of Performance: Theoretical "Perfect" Result


* A perfect **Model A** (the copycat) would output the quantized input, resulting in an MAE equal to the average quantization error of the dataset. **In the dataset used for this experiment, the 1/32 quantization distorted the original length of song by an average of 9.78 ticks, so the theoretical performance upper bound of Model A is about 9.78 ticks.**

    * NOTE: The `9.78 ticks` value represents the **difficulty level** or the **magnitude of the challenge** that Model B has to solve.

      **The Problem:** "Restore the original rhythm from a quantized input that is, on average, 9.78 ticks away from the original."

      **The Difficulty:** To solve this problem perfectly, the model must successfully correct for those **9.78 ticks** of error.

* A perfect **Model B** (the artist) would perfectly reconstruct the original performance, resulting in an MAE of **0 ticks**.

Our goal is to show that our real-world Model B has a significantly lower MAE than Model A, demonstrating that it has successfully learned to apply the singer's rhythmic style.



## Result

after training about 30 epochs

- Final Test MAE for Model A (Control): 10.1670 ticks
- Final Test MAE for Model B (Your Method): 7.6834 ticks

The maximum possible improvement for Model A is only 10.16 - 9.78 = 0.38 ticks. It has almost no room to get better because it has learned the wrong, simple strategy. Task that model A should solve is pretty easy, because it is just identity function.

The maximum possible improvement for Model B is the full 7.68 ticks. We can improve model B so that decrease error 7.68, by increasing model capacity or changing this GRU architecture to more powerful model architecture like Transformer.




## Reproducible Environment with Hugging Face Spaces

The easiest way to run this entire experiment without any local setup is by using the provided Hugging Face Space. It launches a JupyterLab instance in a pre-configured Docker environment with the correct Python version and all dependencies installed.

* **Live Executable Notebook:** [https://huggingface.co/spaces/ccss17/toy-duration-predictor](https://huggingface.co/spaces/ccss17/toy-duration-predictor)

 

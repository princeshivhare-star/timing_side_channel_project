# Timing Side-Channel Analysis of ML Inference Workloads

## Overview

This project studies timing side-channel leakage in machine learning inference. It investigates whether hidden workload properties such as batch size and model depth can be inferred from measured execution time.

## Objective

To demonstrate that inference timing may leak sensitive information about machine learning workloads, which is relevant to the security of heterogeneous computing systems.

## Methodology

Two experiments were conducted:

1. Batch-size leakage experiment: measure inference time while varying batch size
2. Model-depth leakage experiment: measure inference time while varying the number of neural network layers

Average execution time was recorded over repeated runs.

## Results

The results show that inference latency increases with both batch size and model depth. Since these properties can be inferred by observing timing behavior, execution time can act as a side-channel.

## Security Relevance

In tightly coupled CPU–GPU systems, timing behavior may reveal workload structure, execution characteristics, or other hidden properties. This project provides a simple demonstration of how such leakage can arise in ML workloads.

## Future Work

* Extend the attack to GPU-based inference
* Analyze memory-transfer timing leakage
* Evaluate noise injection or masking as possible defenses
* Compare secure and non-secure inference settings

## Tech Stack

* Python
* PyTorch
* Pandas
* Matplotlib

## Run Instructions

pip install -r requirements.txt
python attack.py
python analysis.py

# Awesome-Foundation-Models-for-the-Brain

![](https://img.shields.io/badge/PaperNumber-78-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of **Papers**, **Datasets** and **Code Repositories** for ***Foundation Model for the Brain***. This repository compiles a majority of research works in foundation models in neuroscience, though it may not be fully exhaustive.

<!-- ⭐⭐⭐Our detailed thoughts and review of multi-turn LLMs, including task types, common improvements, and open challenges, are presented in this survey: [**A Survey on Evaluation and Enhancement of Large Language Models Multi-turn Interactions**](https://arxiv.xxx). -->
> If you notice any missing research works or spot inaccuracies, feel free to reach out or open an issue!


## Table of Contents
- [Awesome-Foundation-Models-for-the-Brain](#awesome-foundation-models-for-the-brain)
  - [Table of Contents](#table-of-contents)
  - [Papers](#papers)
  - [Tutorials and Other Resources](#tutorials-and-other-resources)
  - [Datasets and Tools](#datasets-and-tools)
  - [Benchmarks](#benchmarks)
  - [Open Challenges](#Open-Challenges)


<!-- ### Keywords Convention

![](https://img.shields.io/badge/Dataset-blue) New dataset created in the work.

![](https://img.shields.io/badge/Benchmark-red) Benchmark proposed in the work.

![](https://img.shields.io/badge/SFT-blueviolet) SFT used in the work.

![](https://img.shields.io/badge/RL-purple) Reinforcement Learning used in the work.

![](https://img.shields.io/badge/Improved-yellow) Other improvement method(s) used in the work. -->


## Papers

- POYO+: Multi-session, multi-task neural decoding from distinct cell-types and brain regions [[ICLR 2025 spotlight](https://openreview.net/forum?id=IuU0wcO0mo)]
- NDT3: A Generalist Intracortical Motor Decoder [[bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1.abstract)] [[GitHub](https://github.com/joel99/ndt3)]
- Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution [[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/934eb45b99eff8f16b5cb8e4d3cb5641-Abstract-Conference.html)] [[GitHub](https://github.com/colehurwitz/IBL_MtM_model)]
- Multi-X DDM: One Model to Train Them All: A Unified Diffusion Framework for Multi-Context Neural Population Forecasting [[arXiv](https://openreview.net/forum?id=R9feGbYRG7)]
- Foundation model of neural activity predicts response to new stimulus types and anatomy [[bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.21.533548v4)]
- Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity [[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html)] [[GitHub](https://github.com/joel99/context_general_bci)]
- POYO-1: A Unified, Scalable Framework for Neural Population Decoding [[NeurIPS 2023](https://arxiv.org/abs/2310.16046)] [[GitHub](https://github.com/neuro-galaxy/poyo)]
- CEBRA: Learnable latent embeddings for joint behavioural and neural analysis [[Nature 2023](https://www.nature.com/articles/s41586-023-06031-6)] [[GitHub](https://github.com/AdaptiveMotorControlLab/cebra)]
- NDT: Representation learning for neural population activity with Neural Data Transformers [[Neurons, Behavior, Data analysis, and Theory](https://arxiv.org/abs/2108.01210)] [[GitHub](https://github.com/snel-repo/neural-data-transformers)]


## Datasets and Tools

- temporaldata: a Python package that provides advanced data structures and methods to work with multi-modal, multi-resolution time series data [[GitHub](https://github.com/neuro-galaxy/temporaldata)] [[installation](https://temporaldata.readthedocs.io/en/latest/concepts/installation.html)] [[tutorial](https://temporaldata.readthedocs.io/en/latest/concepts/creating_objects.html)] [[documentation](https://temporaldata.readthedocs.io/en/latest/package.html)]
- brainsets: a Python package for processing neural data into a standardized format [[GitHub](https://github.com/neuro-galaxy/brainsets)] [[installation](https://brainsets.readthedocs.io/en/latest/concepts/installation.html)] [[tutorial](https://brainsets.readthedocs.io/en/latest/concepts/using_existing_data.html)] [[documentation](https://brainsets.readthedocs.io/en/latest/package/core.html#)]
- torch_brain: a Python library for various deep learning models designed for neuroscience [[GitHub](https://github.com/neuro-galaxy/torch_brain)] [[installation](https://torch-brain.readthedocs.io/en/latest/concepts/installation.html)] [[documentation](https://torch-brain.readthedocs.io/en/latest/package/data/index.html)]


## Benchmarks

- FALCON: Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark [[NeurIPS 2024 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)] [[GitHub](https://github.com/snel-repo/falcon-challenge)]


## Tutorials and Other Resources

- CoSyNe 2025 tutorial "Transformers in Neuroscience" [[website](https://cosyne-tutorial-2025.github.io)]
- CoSyNe 2025 workshop "Building a foundation model for the brain" [[website](https://neurofm-workshop.github.io)]

## Open Challenges

<!-- In our survey paper on multi-turn interactions and tasks for large language models (LLMs), we categorize a wide range of tasks, including instruction-following scenarios and more complex conversational engagement tasks. To complement this, we also include an illustration highlighting key open challenges in this domain. If you're interested in the detailed improvement methods and a deeper discussion of the open challenges, please refer to our [Full Paper](https://arxiv.xxx).

![](figs/challenges.png) -->





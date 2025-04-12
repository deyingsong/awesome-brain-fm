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
  - [Other related papers](#other-related-papers)
    - [Machine learning and statistical models](#machine-learning-and-statistical-models)
    - [Large-scale experimental recording and imaging](#large-scale-experimental-recording-and-imaging)
    - [Perspective and review](#perspective-and-review)
  - [Benchmarks](#benchmarks)
  - [Open Challenges](#Open-Challenges)


### Keywords Convention

subject:
![](https://img.shields.io/badge/Mouse-blue) Data from mouse.
![](https://img.shields.io/badge/Rat-blue) Data from rat.
![](https://img.shields.io/badge/NHP-blue) Data from non-human primates.
![](https://img.shields.io/badge/Human-blue) Data from human.

recording method:
![](https://img.shields.io/badge/Ephys-green) Electrophysiology recording.
![](https://img.shields.io/badge/Calcium-green) Calcium imaging.
![](https://img.shields.io/badge/fMRI-green) fMRI imaging.

Task:
![](https://img.shields.io/badge/Visual-orange) Visual task or passively looking.
![](https://img.shields.io/badge/Language-orange) Speech processing.
![](https://img.shields.io/badge/Motor-orange) Motor task, such as reaching.
![](https://img.shields.io/badge/Decision-orange) Decision making task, such as 2AFC or context-dependent 2AFC.
![](https://img.shields.io/badge/Navigation-orange) Spatial navigation task.

<!-- ![](https://img.shields.io/badge/Benchmark-red) Benchmark proposed in the work. 

![](https://img.shields.io/badge/SFT-blueviolet) SFT used in the work.

![](https://img.shields.io/badge/RL-purple) Reinforcement Learning used in the work.

![](https://img.shields.io/badge/Improved-yellow) Other improvement method(s) used in the work. --> 


## Papers

- POYO+: Multi-session, multi-task neural decoding from distinct cell-types and brain regions [[ICLR 2025 spotlight](https://openreview.net/forum?id=IuU0wcO0mo)]
- NDT3: A Generalist Intracortical Motor Decoder [[bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1.abstract)] [[GitHub](https://github.com/joel99/ndt3)]
- Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution [[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/934eb45b99eff8f16b5cb8e4d3cb5641-Abstract-Conference.html)] [[GitHub](https://github.com/colehurwitz/IBL_MtM_model)] [[website](https://ibl-mtm.github.io)]
- Multi-X DDM: One Model to Train Them All: A Unified Diffusion Framework for Multi-Context Neural Population Forecasting [[arXiv](https://openreview.net/forum?id=R9feGbYRG7)]
- Foundation model of neural activity predicts response to new stimulus types and anatomy [[bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.21.533548v4)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity [[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html)] [[GitHub](https://github.com/joel99/context_general_bci)]
- POYO-1: A Unified, Scalable Framework for Neural Population Decoding [[NeurIPS 2023](https://arxiv.org/abs/2310.16046)] [[GitHub](https://github.com/neuro-galaxy/poyo)] [[website](https://poyo-brain.github.io)]
- CEBRA: Learnable latent embeddings for joint behavioural and neural analysis [[Nature 2023](https://www.nature.com/articles/s41586-023-06031-6)] [[GitHub](https://github.com/AdaptiveMotorControlLab/cebra)] [[website](https://cebra.ai)]


## Datasets and Tools

- Allen Brain Observatory, visual coding - optical physiology
  - passive behavior, optical physiology, 7 visual areas, gratings, noise, and natural movies
- <mark>temporaldata</mark>: a Python package that provides advanced data structures and methods to work with multi-modal, multi-resolution time series data [[GitHub](https://github.com/neuro-galaxy/temporaldata)] [[installation](https://temporaldata.readthedocs.io/en/latest/concepts/installation.html)] [[tutorial](https://temporaldata.readthedocs.io/en/latest/concepts/creating_objects.html)] [[documentation](https://temporaldata.readthedocs.io/en/latest/package.html)]
- <mark>brainsets</mark>: a Python package for processing neural data into a standardized format [[GitHub](https://github.com/neuro-galaxy/brainsets)] [[installation](https://brainsets.readthedocs.io/en/latest/concepts/installation.html)] [[tutorial](https://brainsets.readthedocs.io/en/latest/concepts/using_existing_data.html)] [[documentation](https://brainsets.readthedocs.io/en/latest/package/core.html#)]
- <mark>CaImAn</mark>: a Python toolbox for large-scale calcium imaging data analysis [[GitHub](https://github.com/flatironinstitute/CaImAn)] [[installation](https://github.com/flatironinstitute/CaImAn/blob/main/docs/source/Installation.rst)] [[tutorial](https://github.com/flatironinstitute/CaImAn/blob/main/demos/notebooks/demo_pipeline.ipynb)] [[documentation](https://caiman.readthedocs.io/en/latest/)] [[talk](https://www.youtube.com/watch?v=rUwIqU6gVvw)]
- NeMoS: a statistical modeling framework optimized for systems neuroscience in Python, currently focusing on the Generalized Linear Model (GLM). [[GitHub](https://github.com/flatironinstitute/nemos)] [[installation](https://nemos.readthedocs.io/en/latest/installation.html)] [[tutorial](https://nemos.readthedocs.io/en/latest/tutorials/README.html)] [[documentation](https://nemos.readthedocs.io/en/latest/api_reference.html)]
- pynapple: a light-weight Python library for neurophysiological data analysis [[GitHub](https://github.com/pynapple-org/pynapple)] [[installation](https://pynapple.org/installing.html)] [[tutorial](https://pynapple.org/user_guide/01_introduction_to_pynapple.html)] [[documentation](https://pynapple.org/index.html)]
- <mark>torch_brain</mark>: a Python library for various deep learning models designed for neuroscience [[GitHub](https://github.com/neuro-galaxy/torch_brain)] [[installation](https://torch-brain.readthedocs.io/en/latest/concepts/installation.html)] [[documentation](https://torch-brain.readthedocs.io/en/latest/package/data/index.html)]
- plenoptic: a Python library for model-based synthesis of perceptual stimuli [[GitHub](https://github.com/plenoptic-org/plenoptic)] [[installation](https://docs.plenoptic.org/docs/branch/main/install.html)] [[tutorial](https://workshops.plenoptic.org/workshops/CSHL-vision-course-2024/branch/main/)] [[documentation](https://docs.plenoptic.org/docs/branch/main/)] [[talk](https://presentations.plenoptic.org)]
- neurosift: a browser-based tool designed for the visualization of neuroscience data with a focus on NWB (Neurodata Without Borders) files, and enables interactive exploration of the DANDI Archive and OpenNeuro online repositories [[GitHub](https://github.com/flatironinstitute/neurosift?tab=readme-ov-file)] [[example](https://nbfiddle.app/?url=https://gist.github.com/magland/dcddee65b7549fbf0b5e142c07ffbed0%23file-neurosift-examples-ipynb)]
- fastplotlib: an expressive plotting library in Python that enables rapid prototyping for large scale exploratory scientific visualization [[GitHub](https://github.com/fastplotlib/fastplotlib)] [[installation](https://fastplotlib.org/ver/dev/user_guide/guide.html)] [[tutorial](https://github.com/fastplotlib/fastplotlib/blob/main/examples/notebooks/quickstart.ipynb)] [[documentation](https://www.fastplotlib.org/ver/dev/)]


## Other related papers

### Machine learning and statistical models

- PopT: Population Transformer: Learning Population-level Representations of Neural Activity [[ICLR 2025 oral](https://openreview.net/forum?id=FVuqJt3c4L)] [[GitHub](https://github.com/czlwang/PopulationTransformer)] [[website](https://glchau.github.io/population-transformer/)]
- Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data [[ICLR 2024](https://openreview.net/forum?id=W8S8SxS9Ng)] [[GitHub](https://github.com/a-antoniades/Neuroformer)] [[website](https://a-antoniades.github.io/Neuroformer_web/)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- GNOCCHI: Diffusion-Based Generation of Neural Activity from Disentangled Latent Codes [[arXiv 2024](https://arxiv.org/abs/2407.21195)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- LDNS: Latent Diffusion for Neural Spiking Data [[NeurIPS 2024 spotlight](https://arxiv.org/abs/2407.08751)] [[GitHub](https://github.com/mackelab/LDNS)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Language-orange) ![](https://img.shields.io/badge/Motor-orange)
- NDT1: Representation learning for neural population activity with Neural Data Transformers [[Neurons, Behavior, Data analysis, and Theory 2021](https://arxiv.org/abs/2108.01210)] [[GitHub](https://github.com/snel-repo/neural-data-transformers)]
- pi-VAE: Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/510f2318f324cf07fce24c3a4b89c771-Abstract.html)] [[GitHub](https://github.com/zhd96/pi-vae)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Rat-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange) ![](https://img.shields.io/badge/Navigation-orange)
- LFADS: Inferring single-trial neural population dynamics using sequential auto-encoders [[Nature Methods 2018](https://www.nature.com/articles/s41592-018-0109-9)] [[GitHub](https://github.com/tensorflow/models/tree/master/research/lfads)] [[Matlab codes](https://lfads.github.io/lfads-run-manager/)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)


### Large-scale experimental recording and imaging

- Diesel2p mesoscope with dual independent scan engines for flexible capture of dynamics in distributed neural circuitry [[Nature Communications 2021](https://www.nature.com/articles/s41467-021-26736-4)] [[data](https://figshare.com/articles/dataset/Diesel2p_mesoscope_with_dual_independent_scan_engines_for_flexible_capture_of_dynamics_in_distributed_neural_circuitry/15163914)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green)
- Distributed coding of choice, action and engagement across the mouse brain [[Nature 2019](https://www.nature.com/articles/s41586-019-1787-x)] [[GitHub](https://github.com/nsteinme/steinmetz-et-al-2019)] [[data](https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406)] ![](https://img.shields.io/badge/Mouse-blue)  ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Visual-orange) ![](https://img.shields.io/badge/Decision-orange) ![](https://img.shields.io/badge/Motor-orange)
- Spontaneous behaviors drive multidimensional, brainwide activity [[Science 2019](https://www.science.org/doi/10.1126/science.aav7893)] [[GitHub](https://github.com/MouseLand/stringer-pachitariu-et-al-2018a)] [[calcium data](https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622/4)] [[ephys data](https://janelia.figshare.com/articles/dataset/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750/4)] ![](https://img.shields.io/badge/Mouse-blue)  ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/calcium-green) ![](https://img.shields.io/badge/Visual-orange) ![](https://img.shields.io/badge/Motor-orange)
- Inception loops discover what excites neurons most using deep predictive models [[Nature Neuroscience 2019](https://www.nature.com/articles/s41593-019-0517-x)] [[GitHub](https://github.com/cajal/inception_loop2019)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex [[Nature Neuroscience 2019](https://www.nature.com/articles/s41593-019-0550-9)] [[GitHub](https://github.com/alleninstitute/visual_coding_2p_analysis)] [[data](http://observatory.brain-map.org/visualcoding)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)


### Perspective and review

- Decoding the brain: From neural representations to mechanistic models [[Cell 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00980-2)]
- A deep learning framework for neuroscience [[Nature Neuroscience 2019](https://www.nature.com/articles/s41593-019-0520-2)]


## Benchmarks

- SENSORIUM: Retrospective for the Dynamic Sensorium Competition for predicting large-scale mouse primary visual cortex activity from videos [[NeurIPS 2024 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html)] [[website](https://www.sensorium-competition.net)]
- FALCON: Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark [[NeurIPS 2024 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)] [[GitHub](https://github.com/snel-repo/falcon-challenge)]
- Brain-Score: 

## Tutorials and Other Resources

- CoSyNe 2025 tutorial "Transformers in Neuroscience" [[website](https://cosyne-tutorial-2025.github.io)]
- CoSyNe 2025 workshop "Building a foundation model for the brain" [[website](https://neurofm-workshop.github.io)]

## Open Challenges

<!-- In our survey paper on multi-turn interactions and tasks for large language models (LLMs), we categorize a wide range of tasks, including instruction-following scenarios and more complex conversational engagement tasks. To complement this, we also include an illustration highlighting key open challenges in this domain. If you're interested in the detailed improvement methods and a deeper discussion of the open challenges, please refer to our [Full Paper](https://arxiv.xxx).

![](figs/challenges.png) -->





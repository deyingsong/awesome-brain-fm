# Awesome-Foundation-Models-for-the-Brain

![](https://img.shields.io/badge/PaperNumber-78-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of **Papers**, **Datasets** and **Code Repositories** for ***Foundation Model for the Brain***. This repository compiles a majority of research works in foundation models in neuroscience, though it may not be fully exhaustive.

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
![](https://img.shields.io/badge/EEG-green) EEG (electroencepholography) recording.

Task:
![](https://img.shields.io/badge/Visual-orange) Visual task or passively looking.
![](https://img.shields.io/badge/Language-orange) Speech processing.
![](https://img.shields.io/badge/Somatosensory-orange) Somatosensory task, such as shape discrimination with whiskering.
![](https://img.shields.io/badge/Olfactory-orange)  Olfactory task.
![](https://img.shields.io/badge/Motor-orange) Motor task, such as reaching.
![](https://img.shields.io/badge/Decision-orange) Decision making task, such as 2AFC or context-dependent 2AFC.
![](https://img.shields.io/badge/WM-orange) Working memory task.
![](https://img.shields.io/badge/Navigation-orange) Spatial navigation task.

Highlights:
![](https://img.shields.io/badge/Mutli_area-FF0000) Data from at least 3 brain areas, or from both cortical and subcortical areas.

<!-- ![](https://img.shields.io/badge/Benchmark-red) Benchmark proposed in the work. 

![](https://img.shields.io/badge/SFT-blueviolet) SFT used in the work.

![](https://img.shields.io/badge/RL-purple) Reinforcement Learning used in the work.

![](https://img.shields.io/badge/Improved-yellow) Other improvement method(s) used in the work. --> 


## Papers

- POYO+: Multi-session, multi-task neural decoding from distinct cell-types and brain regions [[ICLR 2025 spotlight](https://openreview.net/forum?id=IuU0wcO0mo)]
- Foundation model of neural activity predicts response to new stimulus types and anatomy [[Nature 2025](https://www.nature.com/articles/s41586-025-08829-y)] [[GitHub](https://github.com/cajal/foundation)] [[data](https://bossdb.org/project/microns-minnie)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- NDT3: A Generalist Intracortical Motor Decoder [[bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1.abstract)] [[GitHub](https://github.com/joel99/ndt3)]
- Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution [[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/934eb45b99eff8f16b5cb8e4d3cb5641-Abstract-Conference.html)] [[GitHub](https://github.com/colehurwitz/IBL_MtM_model)] [[website](https://ibl-mtm.github.io)]
- Multi-X DDM: One Model to Train Them All: A Unified Diffusion Framework for Multi-Context Neural Population Forecasting [[arXiv](https://openreview.net/forum?id=R9feGbYRG7)]
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

#### Electrophysiology analysis

- LDNS: Latent Diffusion for Neural Spiking Data [[NeurIPS 2024 spotlight](https://arxiv.org/abs/2407.08751)] [[GitHub](https://github.com/mackelab/LDNS)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Language-orange) ![](https://img.shields.io/badge/Motor-orange)
- Most discriminative stimuli for functional cell type clustering [[ICLR 2024](https://arxiv.org/abs/2401.05342v2#)] [[GitHub](https://github.com/ecker-lab/most-discriminative-stimuli)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Visual-orange)
- One-hot Generalized Linear Model for Switching Brain State Discovery [[ICLR 2024](https://openreview.net/forum?id=MREQ0k6qvD)] [[GitHub](https://github.com/JerrySoybean/onehot-hmmglm)] ![](https://img.shields.io/badge/Rat-blue) ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Navigation-orange) ![](https://img.shields.io/badge/Decision-orange) ![](https://img.shields.io/badge/Somatosensory-orange)
- Nonlinear manifolds underlie neural population activity during behaviour [[bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2023.07.18.549575v3)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- GNOCCHI: Diffusion-Based Generation of Neural Activity from Disentangled Latent Codes [[arXiv 2024](https://arxiv.org/abs/2407.21195)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- Preserved neural dynamics across animals performing similar behaviour [[Nature 2023](https://www.nature.com/articles/s41586-023-06714-0)] [[GitHub](https://github.com/BeNeuroLab/2022-preserved-dynamics)] [[data](https://crcns.org/data-sets/motor-cortex/pmd-1)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models [[NeurIPS 2023 spotlight](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html)] [[GitHub](https://github.com/yulewang97/ERDiff)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- Taking the neural sampling code very seriously: A data-driven approach for evaluating generative models of the visual system [[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/458d9f2dd5c7565af60143630dc62f10-Abstract-Conference.html)] [[GitHub](https://github.com/sinzlab/neural-sampling-neurips2023)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Visual-orange)
- Using adversarial networks to extend brain computer interface decoding accuracy over time [[eLife 2023](https://elifesciences.org/articles/84296#content)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- NDT1: Representation learning for neural population activity with Neural Data Transformers [[Neurons, Behavior, Data analysis, and Theory 2021](https://arxiv.org/abs/2108.01210)] [[GitHub](https://github.com/snel-repo/neural-data-transformers)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- Towards robust vision by multi-task learning on monkey visual cortex [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/06a9d51e04213572ef0720dd27a84792-Abstract.html)] [[GitHub](https://github.com/sinzlab/neural_cotraining)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Visual-orange)
- Long-term stability of cortical population dynamics underlying consistent behavior [[Nature Neuroscience 2020](https://www.nature.com/articles/s41593-019-0555-4)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- pi-VAE: Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/510f2318f324cf07fce24c3a4b89c771-Abstract.html)] [[GitHub](https://github.com/zhd96/pi-vae)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Rat-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange) ![](https://img.shields.io/badge/Navigation-orange)
- LFADS: Inferring single-trial neural population dynamics using sequential auto-encoders [[Nature Methods 2018](https://www.nature.com/articles/s41592-018-0109-9)] [[GitHub](https://github.com/tensorflow/models/tree/master/research/lfads)] [[Matlab codes](https://lfads.github.io/lfads-run-manager/)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)
- Cortical population activity within a preserved neural manifold underlies multiple motor behaviors [[Nature Communications 2018](https://www.nature.com/articles/s41467-018-06560-z)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange)


#### Optical physiology analysis


- Exploring Behavior-Relevant and Disentangled Neural Dynamics with Generative Diffusion Models [[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d55170799265c03b37993e02b71b2cc-Abstract-Conference.html)] [[GitHub](https://github.com/BRAINML-GT/BeNeDiff)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) ![](https://img.shields.io/badge/Decision-orange)
- Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data [[ICLR 2024](https://openreview.net/forum?id=W8S8SxS9Ng)] [[GitHub](https://github.com/a-antoniades/Neuroformer)] [[website](https://a-antoniades.github.io/Neuroformer_web/)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) 
- Pattern completion and disruption characterize contextual modulation in the visual cortex [[bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2023.03.13.532473v2.full.pdf)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) 
- Bipartite invariance in mouse primary visual cortex [[bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.03.15.532836v1)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) 
- It takes neurons to understand neurons: Digital twins of visual cortex synthesize neural metamers [[bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.12.09.519708v1.abstract)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) 
- A flow-based latent state generative model of neural population responses to natural images [[NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/84a529a92de322be42dd3365afd54f91-Abstract.html)] [[GitHub](https://github.com/sinzlab/bashiri-et-al-2021)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) 
- Generalization in data-driven models of primary visual cortex [[ICLR 2021 spotlight](https://openreview.net/forum?id=Tp7kI90Htd)] [[GitHub](https://github.com/sinzlab/Lurz_2020_code)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) 
- Stimulus domain transfer in recurrent models for large scale cortical population prediction on video [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/9d684c589d67031a627ad33d59db65e5-Abstract.html)] [[GitHub](https://github.com/sinzlab/Sinz2018_NIPS)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange) 
- Learning a latent manifold of odor representations from neural responses in piriform cortex [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/17b3c7061788dbe82de5abe9f6fe22b3-Abstract.html)] [[GitHub](https://github.com/waq1129/olfactory)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Olfactory-orange) 


#### EEG (electroencepholography) and fMRI analysis

- PopT: Population Transformer: Learning Population-level Representations of Neural Activity [[ICLR 2025 oral](https://openreview.net/forum?id=FVuqJt3c4L)] [[GitHub](https://github.com/czlwang/PopulationTransformer)] [[website](https://glchau.github.io/population-transformer/)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/EEG-green) ![](https://img.shields.io/badge/Language-orange) 
- JGAT: a joint spatio-temporal graph attention model for brain decoding [[arXiv 2023](https://arxiv.org/abs/2306.05286)] [[GitHub](https://github.com/BRAINML-GT/JGAT)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/fMRI-green) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/WM-orange) ![](https://img.shields.io/badge/Motor-orange)
- Brain kernel: A new spatial covariance function for fMRI data [[NeuroImage 2021](https://www.sciencedirect.com/science/article/pii/S1053811921008533)] [[GitHub](https://github.com/waq1129/brainkernel)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/fMRI-green) ![](https://img.shields.io/badge/WM-orange) ![](https://img.shields.io/badge/Decision-orange) ![](https://img.shields.io/badge/Visual-orange)
- Incorporating structured assumptions with probabilistic graphical models in fMRI data analysis [[Neuropsychologia 2020](https://www.sciencedirect.com/science/article/pii/S0028393220301706)] ![](https://img.shields.io/badge/Human-blue) ![](https://img.shields.io/badge/fMRI-green)


### Large-scale experimental recording and imaging

#### Electrophysiology recording

- Conjoint specification of action by neocortex and striatum [[Neuron 2025](https://www.cell.com/neuron/fulltext/S0896-6273(24)00922-X?uuid=uuid%3Ab00b7eef-9b3b-4791-85f9-6fd6ce0cabc7)] [[GitHub](https://github.com/jup36/MatlabNeuralDataPipeline/tree/master/neural_encoding_trial_types_js2p0)] [[data](https://janelia.figshare.com/articles/dataset/Neuropixels_recording_datasets_associated_with_the_manuscript_Conjoint_specification_of_action_by_neocortex_and_striatum_/28025282)] ![](https://img.shields.io/badge/Mutli_area-FF0000) ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange) 
- Stable, chronic in-vivo recordings from a fully wireless subdural-contained 65,536-electrode brain-computer interface device [[bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2024.05.17.594333v2.abstract)] ![](https://img.shields.io/badge/NHP-blue)  ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Visual-orange) ![](https://img.shields.io/badge/Motor-orange)
- Distributed coding of choice, action and engagement across the mouse brain [[Nature 2019](https://www.nature.com/articles/s41586-019-1787-x)] [[GitHub](https://github.com/nsteinme/steinmetz-et-al-2019)] [[data](https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406)] ![](https://img.shields.io/badge/Mutli_area-FF0000) ![](https://img.shields.io/badge/Mouse-blue)  ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Visual-orange) ![](https://img.shields.io/badge/Decision-orange) ![](https://img.shields.io/badge/Motor-orange)
- Spontaneous behaviors drive multidimensional, brainwide activity [[Science 2019](https://www.science.org/doi/10.1126/science.aav7893)] [[GitHub](https://github.com/MouseLand/stringer-pachitariu-et-al-2018a)] [[calcium data](https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622/4)] [[ephys data](https://janelia.figshare.com/articles/dataset/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750/4)] ![](https://img.shields.io/badge/Mutli_area-FF0000) ![](https://img.shields.io/badge/Mouse-blue)  ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/calcium-green) ![](https://img.shields.io/badge/Visual-orange) ![](https://img.shields.io/badge/Motor-orange)



#### Optical imaging

- Functional connectomics spanning multiple areas of mouse visual cortex [[Nature 2025](https://www.nature.com/articles/s41586-025-08790-w)] [[GitHub](https://github.com/AllenInstitute/MicronsFunctionalConnectomics)] [[data](https://bossdb.org/project/microns-minnie)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- Functional connectomics reveals general wiring rule in mouse visual cortex [[Nature 2025](https://www.nature.com/articles/s41586-025-08840-3)] [[GitHub](https://github.com/cajal/microns-funconn-2025)] [[data](https://bossdb.org/project/microns-minnie)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- Cell-type-specific manifold analysis discloses independent geometric transformations in the hippocampal spatial code [[Neuron 2025](https://www.cell.com/neuron/fulltext/S0896-6273(25)00048-0)] [[GitHub](https://github.com/PridaLab/hippocampal_manifolds)] [[data](https://figshare.com/account/home#/projects/234014)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Navigation-orange)
- State-dependent pupil dilation rapidly shifts visual feature selectivity [[Nature 2022](https://www.nature.com/articles/s41586-022-05270-3)] [[GitHub](https://github.com/sinzlab/neuralpredictors)] [[data](https://gin.g-node.org/cajal/Franke_Willeke_2022)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- Diesel2p mesoscope with dual independent scan engines for flexible capture of dynamics in distributed neural circuitry [[Nature Communications 2021](https://www.nature.com/articles/s41467-021-26736-4)] [[data](https://figshare.com/articles/dataset/Diesel2p_mesoscope_with_dual_independent_scan_engines_for_flexible_capture_of_dynamics_in_distributed_neural_circuitry/15163914)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green)
- A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex [[Nature Neuroscience 2019](https://www.nature.com/articles/s41593-019-0550-9)] [[GitHub](https://github.com/alleninstitute/visual_coding_2p_analysis)] [[data](http://observatory.brain-map.org/visualcoding)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)
- Inception loops discover what excites neurons most using deep predictive models [[Nature Neuroscience 2019](https://www.nature.com/articles/s41593-019-0517-x)] [[GitHub](https://github.com/cajal/inception_loop2019)] ![](https://img.shields.io/badge/Mouse-blue) ![](https://img.shields.io/badge/Calcium-green) ![](https://img.shields.io/badge/Visual-orange)


### Perspective and review

- Decoding the brain: From neural representations to mechanistic models [[Cell 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00980-2)]
- Integrating across behaviors and timescales to understand the neural control of movement [[Current Opinion in Neurobiology 2024](https://www.sciencedirect.com/science/article/pii/S0959438824000059)]
- A deep learning framework for neuroscience [[Nature Neuroscience 2019](https://www.nature.com/articles/s41593-019-0520-2)]


## Benchmarks

- SENSORIUM: Retrospective for the Dynamic Sensorium Competition for predicting large-scale mouse primary visual cortex activity from videos [[NeurIPS 2024 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html)] [[website](https://www.sensorium-competition.net)]
- FALCON: Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark [[NeurIPS 2024 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)] [[GitHub](https://github.com/snel-repo/falcon-challenge)]
- Brain-Score: 
- Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity [[NeurIPS 2021 Datasets and Benchmarks Track](https://arxiv.org/abs/2109.04463)] [[website](https://neurallatents.github.io)] ![](https://img.shields.io/badge/NHP-blue) ![](https://img.shields.io/badge/Ephys-green) ![](https://img.shields.io/badge/Motor-orange) ![](https://img.shields.io/badge/Decision-orange)

## Tutorials and Other Resources

- CoSyNe 2025 tutorial "Transformers in Neuroscience" [[website](https://cosyne-tutorial-2025.github.io)]
- CoSyNe 2025 workshop "Building a foundation model for the brain" [[website](https://neurofm-workshop.github.io)]

## Open Challenges



## Acknowledgements

We thank the organizers, speakers, and participants of [CoSyNe 2025 workshop](https://neurofm-workshop.github.io) "_Building a foundation model for the brain_" for highlighting recent progress and offering valuable insights. This [repository](https://github.com/mazabou/awesome-neurofm) also provides references and insights.



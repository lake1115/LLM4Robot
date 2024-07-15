# [LLM4Robot]

==============================

LLM4Robot is a modular high-level library for end-to-end development in embodied AI, which based on [Habitat 2.0](https://arxiv.org/abs/2106.14405). 


## Table of contents
   1. [Installation](#installation)
   1. [Testing](#testing)


## Installation
1. **Preparing conda env**

   Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
   ```bash
   # We require python>=3.9 and cmake>=3.14
   conda create -n llm4robot python=3.9 cmake=3.14.0
   conda activate llm4robot
   ```

1. **conda install habitat-sim**
   - To install habitat-sim with bullet physics
      ```
      conda install habitat-sim=0.2.4 withbullet -c conda-forge -c aihabitat
      ```
      See Habitat-Sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

1. **Create a soft link for habitat-lab**.
   The habitat-lab version currently supported is 0.2.4 
      ```bash
      pip install -e habitat-lab  # install habitat_lab
      ```
1. **Create a soft link for habitat-baselines**.
   The habitat-baselines version currently supported is 0.2.4 
      ```bash
      pip install -e habitat-baselines  # install habitat_baselines
      ```
1. **Download Dataset**.
   It is recommended to download in the habitat default way
    ```bash
    python test/example.py
    ```
1. **Missing basic packages**
   To check and install some dependencies in the Dockerfile


## Testing

1. **Non-interactive testing**: Test the Pick task: Run the example pick task script
    ```bash
    python test/example.py
    ```

## Related Work

This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper, [Large language model as a policy teacher for training reinforcement learning agents](https://www.researchgate.net/profile/Bin-Liu-120/publication/375959604_Large_Language_Model_as_a_Policy_Teacher_for_Training_Reinforcement_Learning_Agents/links/66553e12bc86444c7207452a/Large-Language-Model-as-a-Policy-Teacher-for-Training-Reinforcement-Learning-Agents.pdf)

## Demonstrate

https://github.com/user-attachments/assets/77b456ea-05e2-4ab5-93f8-292e5376cdc4


# Osu Mapper

## Outline
- [Osu Mapper](#osu-mapper)
  - [Outline](#outline)
  - [Overview](#overview)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)
  - [References](#references)

## Overview
This project aims to enhance [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator). Osu Mapper automatically generates osu! standard mode beatmaps by integrating deep learning and reinforcement learning techniques. This project transforms audio into playable beatmaps with minimal manual intervention.

## Directory Structure
Below is a text-based diagram describing the repository structure and the purpose of each component:

```
.
├── agent.py           # Main program for the reinforcement learning agent
├── mapper_env.py      # Implements the beatmap generation environment
├── reward_function.py # Defines the reward function for beatmap generation
├── utils.py           # Utility functions
├── beatmap_model.pt   # Pre-trained model weights
├── configs/           # Configuration files (e.g., inference parameters)
├── audios/            # Audio files for testing
├── osu_files/         # osu! beatmap files
├── Mapperatorinator/  # Code and models related to osu-diffusion
├── requirements.txt   # Python dependency list
└── pyproject.toml     # Project configuration file
```

## Installation
1. Ensure you have Python 3.8 or newer installed.
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the main agent to generate beatmaps:
```sh
python agent.py
```
For inference, adjust the parameters in the configuration files under the `configs/` directory.

For additional diffusion model features, refer to the documentation within the `Mapperatorinator/` directory.

## License
See [LICENSE](LICENSE) for details.

## References
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)
- [osu-diffusion](https://github.com/OliBomby/osu-diffusion)

For more details, please check the individual components' documentation.
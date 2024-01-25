# Cat Generator 

This repo contains code for a pet-project of mine for generating cat images.
I started this project to learn about and understand generative models (and also because I like cats but currently can not own one :( ), and to learn to train them from scratch.
This project is based on the [HuggingFace Diffusion Model course](https://github.com/huggingface/diffusion-models-class) and I followed that course while developing this project.

**NOTE**: This project is for educational purposes and is not allowed to be used for monetary gain.

## Installation

The Cat Generator has been tested using Python `3.9`.
If you are able to get this running on an older version of Python, or the Cat Generator fails to run on a later version, please open an issue and I will look into it.

You can set up your Python envrionment using any method (e.g., Poetry, pipenv, conda, etc.), but please
make sure you have **all** packages from `requirements.txt` installed. 

**NOTE**: I have tested this with `pipenv` and included the relevant `pipenv` files.
If you have success with other methods, please let me know and I can add instructions here!

Once you have all necessary packages installed, you can train and run a demo of the generator.

## Running Demo

I have set up a Gradio demo for those interested in simply running a demo of the generator.
To run the demo, make sure you have all relevant Python packages installed and run the following:

```
python app.py
```

This will start the app up at `http://127.0.0.1:7860`.

## Future Work
# Cat Generator 

This repo contains code for a pet-project of mine for generating cat images.
I started this project to learn about and understand generative models (and also because I like cats but currently can not own one :( ).

## Installation

The Cat Generator has been tested using Python `3.9`.
If you are able to get this running on an older version of Python, or the Cat Generator fails to run on a later version, please open an issue and I will look into it.

You can set up your Python envrionment using any method (e.g., Poetry, pipenv, conda, etc.), but please
make sure you have **all** packages from `requirements.txt` installed. 

**NOTE**: I have tested this with `pipenv` and included the relevant `pipenv` files.
If you have success with other methods, please let me know and I can add instructions here!

Once you have all necessary packages installed, you can train, evaluate, and run a demo of the generator.

## Running Demo

**Currently under construction**

I have set up a Gradio demo for those interested in simply running a demo of the generator.
To run the demo, make sure you have all relevant Python packages installed and run the following:

```
python app.py
```

This will start the app up at `http://127.0.0.1:7860`.

## Training

```
python train.py
```
For a list of possible hyperparameters that can be tuned, you can add `-h` as an argument:
```
python train.py -h
```

## Evaluation

```
python eval.py
```

TODO: Write evaluation instructions

## Future Work
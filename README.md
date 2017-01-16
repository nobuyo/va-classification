# va-classification
voice actor faces classification

## Environment

- keras(tensorflow backend)
- python3.5 with anaconda3 via pyenv

## Get Started

1. download dataset from [va-face](https://bitbucket.org/nobuyo/va-face)
2. preprocessing the images 

    `python3 pp_images.py`

3. training

    `python3 vac_train.py`
    
4. validate
    
    `python3 vac_test.py --model model-name`

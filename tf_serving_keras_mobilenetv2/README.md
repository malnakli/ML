# Training and Deploying A Deep Learning Model in Keras MobileNet V2 and Heroku

This repo is part of this [blog](https://medium.com/@malnakli/tf-serving-keras-mobilenetv2-632b8d92983c) 

## Installation 
- [conda](https://conda.io/docs/installation.html)

## Run
```
git clone https://github.com/malnakli/ML.git
cd ML/tf_serving_keras_mobilenetv2

conda env create  -f environment.yml 
source activate ml

# or if you don't want to create new conda environment. run the following:
conda env update -n base  -f environment.yml  # 
source activate base

jupyter-notebook main.ipynb
```
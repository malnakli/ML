# Train Keras MobileNet V2 model & Deploy it to Heroku by using Tensorflow Serving
This a complete code for my following blog []().


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
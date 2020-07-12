# kaggle-house-prices
Kaggle competition on house prices prediction


# service-helmet-detector

In this project, we want to training house-prices regression data
## Get started

<details>
<summary>1. If windows.</summary>

```markdown
## Step 1: Add PROJECT_PATH to your environment
$ setx /m PROJECT_PATH <PROJECT_PATH>

## Step 2: Install the python package
#### CPU version
$ pip install -r requirements.txt

## Step 3: Change the config yaml file.

## Step 4: Run the service pipeline.
$ python main.py --mode etl
$ python main.py --mode train_eval --model lr
$ python main.py --mode train_eval --model gbr
```
</details>

<details>
<summary>2. If linux.</summary>

```markdown
## Step 1: Add PROJECT_PATH to your environment.
$ export PROJECT_PATH=/home/app/workdir

## Step 2: Install the python package.
$ pip3 install -r requirements.txt

## Step 3: Change the config yaml file.

## Step 4: Run the service pipeline.
$ python main.py --mode etl
$ python main.py --mode train_eval --model lr
$ python main.py --mode train_eval --model gbr
```
</details>

## Version, author and other information:
- See the relation information in [setup file](setup.py).

## License
- See License file [here](LICENSE).
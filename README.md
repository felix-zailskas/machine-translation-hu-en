## Creating the Environment

To create the virtual environment in this project you must have `pipenv` installed on your machine. Then run the following commands:

```
# for development environment
pipenv install --dev
# for production environment
pipenv install
```
To work within the environment you can now run:

```
# to activate the virtual environment
pipenv shell
# to run a single command
pipenv run <COMMAND>
```

## Running the Application

For ease of use this repository contains a collection of python scripts that can be executed to read in and preprocess the data, create word embeddings, as well as training and testing models. These can be found in the `src/scripts/` directory. Make sure to extract the `hu-en.tgz` file in the `data/` directory. All scripts should be executed from the main directory due to relative paths within them. To create different experiments adjust the parameters defined at the top of these scripts and in the `src/utils/constants.py` file.<br><br>

The scripts must be executed in the following order:

### 1. `read_data.py`
This script will read the data from the raw text files in the `data/hu-en/` directory and parse them into a CSV file stored under `data/sampled_data.py`. Adjust the `sample_proportion` and `offset` parameters to get different parts of the raw data.

### 2. `preprocess_data.py`
This script preprocess the data with standard NLP techniques and truncate the data to sentences containing at most `src/utils/constants.MAX_WORDS` words. The data is then split into training, validation and test set, according to the set proportions.

### 3. `create_embeddings.py`
This script will create all needed word embeddings for the experiments.

### 4. `train_word_based_model.py`
This script will train and save a model. The model that should be trained can be defined through the parameters in the script.

### 5. `test_model.py`
This script will test the defined model using the BLEU score and display relevant statistics and plots.

## Contribution Workflow

This repository uses `pre-commit` hooks to ensure a consistent and clean file organization. Each registered hook will be executed when committing to the repository. To ensure that the hooks will be executed they need to be installed using the following command:

```
pre-commit install
```

The following hooks are registered in this Project:

<ul>
<li>
<b>Black:</b> Black is a PEP 8 compliant opinionated formatter. Black reformats entire files in place.
</li>
</ul>

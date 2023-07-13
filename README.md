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
<li>
<b>isort:</b> isort is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type
</li>
</ul>

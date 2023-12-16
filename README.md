# Data Engineering Toolkit
A Package to help handel typical Data Engineering Tasks Handel Typical Data Engineering Tasks, including: 
* Storage
* Formatting
* Tagging / Labelling
* Profiling
* Cleaning
* Transformations
* Bussiness Rules
* Encryption


## Status of the project
current_version = "2.0.0"

The Data Engineering Toolkit is still under initial development and is being tested with Python 3.11.4 version.

The Data Engineering Toolkit will follow semantic versioning for its releases, with a `{major}.{minor}.{patch}` scheme for versions numbers, where:

* `major` versions might introduce breaking changes
* `minor` versions usually introduce new features and might introduce deprecations
* `patch` versions only introduce bug fixes


## Replit development
```shell
poetry lock --no-update
poetry install -E dev
```

## Local Setup

### Windows
* Open new terminal
    * "Windows-Key + R" will show you the 'RUN' box
    * Type "cmd" to open the terminal
```shell
cd <Path To>/DataEngineeringToolkit

python -m venv venv

venv\Scripts\activate

```
### Linux / Mac
* Open new terminal
    * "Control + Option + Shift + T" to open the terminal
```shell
cd <Path To>/DataEngineeringToolkit

python -m venv venv

source venv/bin/activate

```

### Local development
```shell
(venv) python -m pip install -r requirements.txt

(venv) poetry lock --no-update
(venv) poetry install -E dev
```

## Build Package
```shell
(venv) bumpver update --minor
(venv) python -m build
```
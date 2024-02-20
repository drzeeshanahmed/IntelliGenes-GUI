# IntelliGenes Desktop
IntelliGenes desktop is a user-friendly desktop wrapper around the
[IntelliGenes CLI](https://github.com/drzeeshanahmed/intelligenes).

## Building Source
To build the application from source, first clone this repository and then
follow these steps:

### 1. Setup Virtual Environment
To install pip packages, it is recommended to optioanlly set up a virtual environment.
This helps reduce the overall bundle size.
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install requirements.txt
```

### 3. Bundle script
You should be able to execute the following command for a fast build:
```bash
pyinstaller main.spec --noconfirm
```
To build from scratch (first time), you can instead run:
```bash
pyinstaller main.py -D --noconsole --noconfirm --collect-all "xgboost"
```
Note, the `--collect-all "xgboost"` flag is required because xgboost has some
hidden runtime imports that need to be included in the bundle.

### 4. Run Application
Inside the generated `dist` folder, there should exist an executable for your
current architecture. Launch that application to use IntelliGenes Desktop.
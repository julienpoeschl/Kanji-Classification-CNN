# Development

## Create venv from requirements.txt
In project root directory:
```bash
pip install -r requirements.txt
```

## Run application
In project root directory:
```bash
python app/src/main.py
```

## Create dataset
In project root directory:
```bash
python dataset/src/dataset_generation.py
python dataset/src/dataset_preprocessing.py
```

## Train model
In project root directory:
```bash
python model/src/training.py
```

## Create local distro of application
In project root directory:
```bash
pyinstaller --onefile --windowed app/src/main.py
pyinstaller app/src/main.py 
pyinstaller main.spec
```

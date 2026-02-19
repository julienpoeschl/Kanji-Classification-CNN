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


## Get kanji data from https://www.kanjidatabase.com/index.php
```sql
SELECT
  `Kanji`,
  `Strokes`,
  `Grade`,
  `Kanji Classification`,
  `JLPT-test`,
  `Reading within Joyo`,
  `Reading beyond Joyo`,
  `On within Joyo`,
  `Kun within Joyo`,
  `Translation of On`,
  `Translation of Kun`,
  `# of Meanings of On`,
  `# of Meanings of Kun`
FROM KANJI6654
WHERE Grade IS NOT NULL
  AND `JLPT-test` IN ('5','4','3','2')
ORDER BY `Kanji Frequency without Proper Nouns` DESC
LIMIT 1000
```

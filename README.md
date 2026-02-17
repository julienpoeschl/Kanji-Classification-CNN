# Kanji-Classification-CNN
A CNN model pretrained to label pictures of arbitrary size to japanese kanji characters implemented in PyTorch.

### Dataset
[Generated](https://github.com/julienpoeschl/Kanji-Classification_CNN/dataset/src/data_generation.py) using these free fonts:
- AoyagiSosekiFont2 found [here](https://www.freekanjifonts.com/japanesefont-aoyagisoseki-download/) 
- ipamjm found [here](https://github.com/ken1row/IPSJ-techrep-xelatex) with [license](https://github.com/julienpoeschl/Kanji-Classification_CNN/dataset/data/fonts/licenses/IPAフォントライセンスv1.0.txt)
- JiyunoTsubasa found [here](https://www.freekanjifonts.com/japanesefont-jiyuno-tsubasa/)
- [KikaiChokokuJISTTF](https://font.kim) found [here](https://www.freejapanesefont.com/kikai-chokoku-jis-font-download/)
- KouzanMouhituFontOTF found [here](https://www.freekanjifonts.com/japanesefont-kozan-mohitsu-download/)
and a list of the 1000 most used kanji from [https://www.kanjidatabase.com/index.php](https://www.kanjidatabase.com/index.php) (and additional data per kanji) using this query:
```sql
SELECT `Kanji`, `Strokes`, `Grade`, `Kanji Classification`, `JLPT-test`, `Reading within Joyo`, `Reading beyond Joyo`, `On within Joyo`, `Kun within Joyo`, `Translation of On`, `Translation of Kun`,`# of Meanings of On`, `# of Meanings of Kun`
FROM KANJI6654
WHERE Grade IS NOT NULL
  AND `JLPT-test` IN ('5','4','3','2')
ORDER BY `Kanji Frequency without Proper Nouns` DESC
LIMIT 1000
```

The dataset is augmented dynamically during training (slight rotation, shifts, color inversion, ...).


### Model
- PyTorch
- CNN
- Classification

Using device: cpu
Loading processed data...
Loaded 5000 samples with 1000 classes
Input shape: (5000, 64, 64, 1)
Preparing data loaders...
Train batches: 7032, Val batches: 8
Creating model...
Total parameters: 1,818,184
Trainable parameters: 1,818,184

### Application
- Send screenshot to evaluate
- Read result + confidence level

### Evaluation
- Accuracy during training + validation
- 

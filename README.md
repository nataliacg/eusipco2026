# EUSIPCO 2026 – Bioacoustic Model Comparison



This repository contains the experimental code and dataset manifests used for the EUSIPCO 2026 submission.



The project compares different bioacoustic foundation models under zero-shot and fine-tuning settings:



\- BioLingual (CLAP-based)

\- NatureLM (Earth Species Project)

\- BirdNET (BirdNET-Analyzer)



---



## Repository Structure

eusipco2026/
│
├── data/
│
└── manifests/        # CSV files defining dataset splits
│
├── src/
│   
├── zero\_shot/        # Zero-shot evaluation scripts
│   
└── finetune/         # Fine-tuning / MLP adaptation scripts
│
├── requirements.txt
└── .gitignore

---



## Dataset



This repository \*\*does not include raw audio files\*\*.



Only CSV manifests are provided.  

Each manifest contains at least:



\- `clip\_path`

\- `true\_scientific\_name`



Audio files must be available locally following the paths defined in the manifest.



---



## Environment Setup



We recommend using separate virtual environments for each model:



### Example (Windows)



```bash

python -m venv .venv

.\\.venv\\Scripts\\activate

pip install -r requirements.txt

```


Model Weights



Model weights are not included in this repository.



BioLingual / CLAP



Requires local model directory passed via --model\_path.



NatureLM



Loaded directly from HuggingFace:



&nbsp;	NatureLM.from\_pretrained("EarthSpeciesProject/NatureLM-audio")

BirdNET



Requires installing the official BirdNET-Analyzer:



https://github.com/kahst/BirdNET-Analyzer



BirdNET experiments should be run inside the BirdNET environment.









Running Zero-Shot Experiments

```bash

python src/zero\_shot/biolingual.py \\

    --manifest data/manifests/dataset\_manifest\_TEST\_9.csv \\

    --model\_path /path/to/biolingual\_model \\

    --results\_dir results/biolingual/test\_9

```



```bash
python src/zero\_shot/naturelm\_audio.py \\

    --manifest data/manifests/dataset\_manifest\_TEST\_9.csv \\

    --results\_dir results/naturelm/test\_9


```

```bash
src/zero\_shot/birdnet.py \\

    --manifest data/manifests/dataset\_manifest\_TEST\_40.csv \\

    --results\_dir results/birdnet/test\_40


```


Outputs



Each experiment produces:

&nbsp;	•	classification\_report.csv

&nbsp;	•	confusion\_matrix.csv

&nbsp;	•	predictions.csv

&nbsp;	•	summary.txt



Generated outputs are not tracked by git.





Notes

&nbsp;	•	Audio data is not distributed in this repository.

&nbsp;	•	API keys (if required) must be provided via environment variables.

&nbsp;	•	Virtual environments are not included.




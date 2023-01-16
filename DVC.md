# List of DVC assets

- Knowledge Base input files
  (- Knowledge Base )
  [kb_clean_merge_datasets.py, make_kb.py]
- Enriched candidate data
- Input data
- Annotation data
  
- Spacy config
- Spacy meta.json
- Spacy model
  [prepare_training.py]

# Pipeline

- Make KB from input files (includes cleaning)
- Convert KB and text datasets into spaCy Docs
- Trigger spaCy model training


## Annotation data

Annotation data is gathered on a cloud server that runs Prodigy in order to allow different annotators to connect to and annotate the same dataset.

In order to back up and track changes to this dataset, we followed this [forum post](https://support.prodi.gy/t/prodigy-and-dvc-data-version-control/3390)
and created a `cron` job that regularly outputs the dataset and commits it to DVC.

```crontab
45 */2 * * * python -m prodigy db-out el_session_1 </path/to/repo>/prodigy/data/ && dvc commit -m "Automatic backup `date +\%Y-\%m-\%dT\%H:\%M:\%S`" </path/to/repo>/prodigy/data/
```

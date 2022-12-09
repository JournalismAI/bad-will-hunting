# Custom Entity Linking Prodigy Recipe 

This folder contains the custom recipe and configuration for running the JAI 2022 entity linking candidate annotation.

## How to run?
Please ensure that you run this code in the same directory in order to allow Prodigy to pick up the local `prodigy.json` file.
Otherwise Prodigy will fall back to the default system `prodigy.json` and none of the customisations will be reflected in the UI.

To start the annotation server, run the following:
```bash
 python -m prodigy entity_linker.manual <dataset name> <source_file> <nlp_model> <kb_file> <additional_info_entities_file> -F el_recipe.py
```

<dataset name> - custom user input
<source_file> - provided as jsonl. An example can be in prodigy/data/paragraph_text_mentions.jsonl. 
<nlp_model> - default or a custom spaCy model
<kb_file> - KB directory generated in the /data folder by the dvc pipeline
<addiotional_info_entities_file> - .csv generated in the /data folder by the dvc pipeline

To display the command reference on the CLI, run
```bash
prodigy entity_linker.manual --help -F el_recipe.py
```

For further information, see the [Prodigy documentation](https://prodi.gy/docs/custom-recipes) pages. 
# README

## Directory
This Github contains sample code for Xenium lung cancer dataset.

```
public_project/
|-- experiments/
|   |-- config/
|   |   |-- {config}.yaml # config
|   |-- data/
|   |   |-- # test data (spatial transcriptome data, reference data of marker genes, colormap for visualization)
|   |-- notebooks/
|   |   |-- run_experiments.ipynb
|   |-- runs/
|   |   |-- # results 
|-- src/
|   |-- configs/
|   |-- evaluation/
|   |-- io/
|   |-- preprocess/
|   |-- steps/
|   |   |-- step1/ 
|   |   |-- step2/
|-- README.md
```


## Input & Output
Input　`data/`
- Spatial transcriptome data: N x 3 
    - "local_pixel_x", "local_pixel_y" (float) : xy coordinate of transcripts. (pixel)
    - "gene" (str) : gene name
- Reference data:
    - key > celltype (str)
    - value > "neg_markers" (list[str]), "pos_markers" (list[str]) : negative/positive marker gene name list

Output `runs/{dataset_name}/`
- `{timestamp}_step1_outputs/`: cell type region estimation results
- `{timestamp}_step2_outputs/`: cell region estimation results
- `{timestamp}_eval_outputs/`: evaluation metrics
- .json: colormap for visualization reproducibility
- .yaml: timestamp 

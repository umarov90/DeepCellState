## DeepCellState: drug response prediction
![Framework](framework.png)
Drug treatment induces cell type specific transcriptional programs, and as the number of combinations of drugs and cell types grows, the cost for exhaustive screens measuring the transcriptional drug response becomes intractable. We developed DeepCellState, a neural network auto-encoder framework based on the DeepFake approach, for prediction of the induced transcriptional state in a cell type after drug treatment, based on the drug response in another cell type.

To run DeepCellState, download the [data](https://www.dropbox.com/s/dluxw8zryh1hoyf/DeepCellState_data.zip?dl=1) and extract it to some location. Add this path to to the data_dir file in the project root. 
Run train_models.py to generate the models. 

Trained models can be downloaded from:
[models](https://www.dropbox.com/s/7c77tzxaefhom2d/DeepCellState_models.zip?dl=1)

DeepCellState requires ```tensorflow>=2.3.0```, which can be installed via pip:
```sh
pip install tensorflow
```
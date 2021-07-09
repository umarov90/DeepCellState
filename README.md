## DeepCellState: drug response prediction
![Framework](framework.png)
Drug treatment induces cell type specific transcriptional programs, and as the number of combinations of drugs and cell types grows, the cost for exhaustive screens measuring the transcriptional drug response becomes intractable. We developed DeepCellState, a neural network auto-encoder framework, for prediction of the induced transcriptional state in a cell type after drug treatment, based on the drug response in another cell type.

The implementation details can be found in the [paper](https://doi.org/10.1101/2020.12.14.422792).

DeepCellState requires ```tensorflow==2.3.0```, which can be installed via pip:
```sh
pip install tensorflow==2.3.0
```

Also, please install the following packages:
```sh
pip install pandas==1.0
pip install matplotlib
pip install seaborn
```

To run DeepCellState, download the [data](https://www.dropbox.com/s/merj99vfp4fpdg2/DeepCellState_data.zip?dl=1) ([Mirror](https://drive.google.com/file/d/1JV-6xYl-uMswzIP3-dhmiG-LNFmiCXhx/view?usp=sharing)) and extract it to some location. Add this path (**parent folder of the data folder**) to the 'data_dir' file in the repository root, after you clone the repository to your machine. 
For example:

/home/user/Desktop/deepcellstate_test/   
(please have '/' at the end) 

Run DeepCellState.py to generate the models (takes 1-3 hours depending on your GPU):
```sh
python3 DeepCellState.py -O models_output -CT A375,HEPG2 -PT trt_cp -N 1 -SM 0
```
Parameters:
- ```-O```: Output directory.
- ```-CT```: Comma separated list of cell types to use in addition to MCF7 and PC3.
- ```-PT```: Perturbation type to be used, defaults to trt_cp. 
- ```-N```: Number of models trained for each fold. The model with best validation performance is picked.
- ```-SM```: Set to 1 to train drug MoA family models or set to 2 to train external validation model. Defaults to 0, i.e. 10-fold cross-validation.

After training is complete, the output directory will contain the new models and text files with the performance.

Trained models can be downloaded from:

[models 1](https://www.dropbox.com/s/9b0l6zczdjp28im/DeepCellState_model_ext_val.zip?dl=1) ([Mirror](https://drive.google.com/file/d/1zD44n2dd7-jMJqCRYSRSxPHeR7Rn-lgx/view?usp=sharing)) (external validation and analysis model. Decoders: HEPG2, MCF7, PC3)

[models 2](https://www.dropbox.com/s/02ibrv4lasye36i/DeepCellState_models_2cells.zip?dl=1) ([Mirror](https://drive.google.com/file/d/1_KbeF7_euic4v6UnYCThq34_REnBjz6-/view?usp=sharing)) (two cell types 10-fold cross-validation models. Decoders: MCF7, PC3)

[models 3](https://drive.google.com/file/d/1SHHTXpJBZoBhwqK0vvlw9bmwhPv16K3n/view?usp=sharing) (all the models used in the paper)


Extract them into the same directory as the data folder. All ext_val.py, ext_val_cancer.py, tf_analysis.py, and gene_importance_analysis.py require best_autoencoder_ext_val (link 'models 1' above) to be in the *data_dir* folder.
To reproduce external data validation performed in the paper, download the [raw data](https://drive.google.com/file/d/1uZReFhhAXmudAyEt4lSX-of2CI2d1eYv/view?usp=sharing) and place it in the *data_dir*/data/ folder.
Run ext_val.py for the statin data and ext_val_cancer.py for the anti-cancer data. The above-mentioned scripts generate output in the 'figures_data' folder which can be visualized by running scripts in the 'figures' folder of this repository. 
The produced images will be placed in the 'figures' folder inside the specified *data_dir* folder. 
To generate profile figures during training, uncomment the lines in DeepCellState.py, please note that it will make training slower. 

DeepCellState was implemented and tested on Ubuntu 18. Please contact us if you have any issues running the code!

The input and output for FaLRTC which was used in the manuscript can be downloaded from:

[FaLRTC](https://drive.google.com/file/d/1-h5xduucFhQ_Skd21GPwsObQptdkiqBQ/view?usp=sharing)

This is how DeepCellState models can be used in a custom script to make predictions:

```python
import pickle
from tensorflow import keras

model = "best_autoencoder_ext_val/"
autoencoder = keras.models.load_model(model + "main_model/")
cell_decoders = {"MCF7": pickle.load(open(model + "MCF7" + "_decoder_weights", "rb")),
                 "PC3": pickle.load(open(model + "PC3" + "_decoder_weights", "rb"))}
autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])
pc3_response = ... # get an array of dimension 1, 978, 1
# convert PC3 response to MCF7 response
mcf7_predicted_response = autoencoder.predict(pc3_response) 
```

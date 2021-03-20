## DeepCellState: drug response prediction
![Framework](framework.png)
Drug treatment induces cell type specific transcriptional programs, and as the number of combinations of drugs and cell types grows, the cost for exhaustive screens measuring the transcriptional drug response becomes intractable. We developed DeepCellState, a neural network auto-encoder framework, for prediction of the induced transcriptional state in a cell type after drug treatment, based on the drug response in another cell type.

The implementation details can be found in the [paper](https://doi.org/10.1101/2020.12.14.422792).

DeepCellState requires ```tensorflow==2.3.0```, which can be installed via pip:
```sh
pip install tensorflow==2.3.0
```

Please also make sure you have pandas 1.0 installed:
```sh
pip install pandas==1.0
```

To run DeepCellState, download the [data](https://www.dropbox.com/s/merj99vfp4fpdg2/DeepCellState_data.zip?dl=1) ([Mirror](https://drive.google.com/file/d/1lGnUANHpKU33pEvl7meEEVMG7wJM1Th1/view?usp=sharing)) and extract it to some location. Add this path (**parent folder of the data folder**) to the 'data_dir' file in the repository root, after you clone the repository to your machine. 
For example:

/home/user/Desktop/deepcellstate_test/   
put '/' at the end. 

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

[models 1](https://www.dropbox.com/s/7c77tzxaefhom2d/DeepCellState_models.zip?dl=1) ([Mirror](https://drive.google.com/file/d/14__66BjjDTyB19p0NbK3r_eNOujV_60Z/view?usp=sharing)) (two cell types and external validation)

[models 2](https://drive.google.com/file/d/1SHHTXpJBZoBhwqK0vvlw9bmwhPv16K3n/view?usp=sharing) (all models used in the paper)


Extract them into the same directory as the data folder. All ext_val.py, ext_val_cancer, tf_analysis.py, and gene_importance_analysis.py require best_autoencoder_ext_val to be in the same folder as the data folder. 

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

Figures from the paper can be recreated by running scripts in the 'figures' folder. Please install matplotlib and seaborn first:
```sh
pip install matplotlib
pip install seaborn
```

To generate profile figures during training, uncomment the lines in DeepCellState.py.

DeepCellState was implemented and tested on Ubuntu 18. Please contact us if you have any issues running the code!
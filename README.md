# Data-Challenge-2

This is the algorithmic policing tool for the course Data Challenge 2 [JBG050] of group 20. In this file you can read how to use the tool and answer some questions that might come up while using the tool. The tool might have problems finding some folders when run on something else than Windows, in this case, the user might have to replace the paths to some files.

## Files

### Code
The main files in the repository are:
- `Final dashboard.ipynb` -> this is the dashboard code
- `Pre-processing.ipynb` -> this is the tool to pre-proccess the data that we have used to visualize our tool
- `getBestModels.py` -> This file finds the best models and exports the `best_models.csv` file
- `datasets` -> This is a folder that contains all required data for the tool and preprocessing to run

### Data
In `datasets` all of the following files are stored:
- `train.zip` -> This is a zip file which contains a .csv file of `train.csv`
- `test.zip` -> This is a zip file which contains a .csv file of `test.csv`
- `complete.zip` -> This is a zip file which contains a .csv file of both the contents of `train.csv`, `test.csv` and more recent data
- `MSOA-Names.csv` -> This file contains all of the MSOA names with their corresponding LSOA number.
- `best_models.csv` -> This file contains the parameters for the ARIMA models, and also the AIC and MASE from the ARIMA models. This file is exported by `getBestModels.py`

**The zip files within the `datasets` folder should be unzipped first before they can be used.**

The data provided by the course is also needed. Only the `street.csv` files are needed to run the dashboard tool, but these should be automatically found by the tool when provided with the right path to these files. As these files are 1) too large and 2) too many to store on GitHub, these could not be uploaded to GitHub. Therefore, the user must download these themselves via the link from the handbook of the course.

The .ipynb files can be used in any preferred way to run the .ipynb files, which are mainly Jupyter Notebooks. However, they can be exported to a .py file to run them in any IDE that supports .py files to the users desires.

## Running the code
### Dependencies
In both the `Pre-processing.ipynb` and `Final dashboard.ipynb` in the beginning a import section of libraries has been given. All of the stated libraries have to be installed before they can be imported. This is crucial for the tool to run in a proper way.

While installing the GeoPandas library you might have some trouble. This can easily be fixed by creating a new environment and installing needed libraries as needed in the dashboard.

### Pre-processing
Whenever you want to run the `Pre-processing.ipynb` to get the output of `train.csv`, `test.csv`, and `complete.csv` yourself, that is possible. You can simply open the file and run it. When running, you need to have the `MSOA-Names.csv` file ready in the `datasets` folder, and the tool will ask you to provide the location of the original dataset folder (containing the street files (the other files do not need to be filtered out however)). The original dataset is not on GitHub, and has to be provided by the user. Whenever you run the file, after a while, the `train.csv`, `test.csv`, and `complete.csv` files will get exported to the `datasets` folder (which should already exist) and be located where `Pre-processing.ipynb` is located. 

### Generating the best models
Whenever you want to run the `getBestModels.py` to get the output of `best_models.csv` yourself, that is possible as well. Running this script takes quite an amount of time, so this is not recommended.

### Visualising results
If you want to use the visualization tool, you can run `Final dashboard.ipynb` in the same way as `Pre-processing.ipynb`. It must be fully run, before you can see the dashboard. It will ask you to first provide the path to the dataset provided by the course, and second to provide a path to the `datasets` folder within this project. However, if you are running on a different OS than Windows, you might have to add the paths to these folders directly in the code. Whenever you run the file, it takes a while for the local visualization tool server to start. This can take around a couple of minutes, however, it might be faster depending on the specification and performance of your machine. When the tool is ready to open, you can open http://127.0.0.1:8050/ in your browser, or click the link in the 'run' tab of your IDE or Jupyter Notebook. 

## The tool
In the tool, the following elements can be found:
- In the top a tab selector can be seen. This makes it easy for the user to switch between different views (Analysis (comparison and hotspot-analysis), and forecasting).
- On the left a widget screen with filters where different conditions and variables can be selected.
- In the middle the visualisation of the selected tab, condition and variables.

Hopefully everything is clear and the tool is ready to be used!

# Data-Challenge-2

This is the algorithmic policing tool for the course Data Challenge 2 [JBG050] of group 20. In this file you can read how to use the tool and answer some questions that might come up
while using the tool.

The main files in the repository are:
- 'Final dashboard.ipynb’ -> this is the dashboard code
- 'Pre-processing.ipynb’ -> this is the tool to pre proccess the data that we have used to visualize our tool
- ‘getBestModels.py’ -> This file exports the ‘best_models.csv’ file
- ‘datasets’ -> This is a folder that contains all required data for the tool and preprocessing to run
 
In ‘datasets’ all of the following files are stored:
- ‘train.zip’ -> This is a zip file which contains a .csv file of ‘train.csv’
- ‘test.zip’ -> This is a zip file which contains a .csv file of ‘test.csv’
- ‘complete.zip’ -> This is a zip file which contains a .csv file of both the contents of ‘train.csv’ and ‘test.csv’
- ‘MSOA-Names.csv’ -> This file contains all of the MSOA names with their corresponding LSOA number.
- ‘best_models.csv’ -> This file contains the parameters for the ARIMA models, the AIC and MASE from the ARIMA model. This file is exported by ‘getBestModels.py’

The zip files within the ‘datasets’ folder should be unzipped first before they can be used

The .ipynb files can be used in any preferred way to run the .ipynb files, which are mainly Jupyter Notebooks. However, they can be exported to a .py file to run them in any IDE that supports .py files to the users desires.

In both the ‘pre-processing.ipynb and 'Final dashboard.ipynb’ in the beginning a import section of libraries has been given. All of the stated libraries have to be installed before they can be
imported. This is crucial for the tool to run in a proper way.

Whenever you want to run the ‘pre-processing.ipynb’ to get the output of ‘train.csv’, ‘test.csv’, and ‘complete.csv’ yourself, that is possible. You can simply open the file, however, before you 
do so something has to be changed. The directory of the folder where all of the original dataset (as provided by the course (this could not be fit only GitHub) has to get changed according to the directory on your machine. Whenever you run the file, after a while, the ‘train.csv’, ‘test.csv’, and ‘complete.csv’  files will get exported in the same folder as where ‘pre-processing.ipynb’ is located. 

Whenever you want to run the ‘getBestModels.py’ to get the output of ‘best_models.csv’ yourself, that is possible as well. The user has to change the directories of the ’train.csv’ and ‘test.csv’ file accordingly. Running this script takes quite an amount of time, so this is not recommended.


If you want to use the visualization tool, you can run 'Final dashboard.ipynb’ in the same way as ‘pre-processing.ipynb’. However, one again, you have to change some lines that contain 
local directories.  Whenever you run the file, it takes a while for the local visualization tool server to start. This takes around a couple of minutes, however, it might be faster depending on the specification and performance of your machine. When the tool is ready to open, you can open http://127.0.0.1:8050/ in your
browser, or click the link in the 'run' tab of your IDE or Jupyter Notebook. 


In the tool, the following elements can be found:
- In the top a tab selector can be seen. This makes it easy for the user to switch between different views (Analysis (comparison and hotspot-analysis), and forecasting)
- On the left a widget screen where different conditions and variables can be selected
- In the middle the visualisation of the selected tab, condition and variables

I hope everything is clear and the tool is ready to be used!

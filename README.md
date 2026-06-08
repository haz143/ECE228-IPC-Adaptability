# IPC-prediction
**Gem5-Mimic Machine Learning Project**

Overview:
This GitHub repository contains a project that uses machine learning models to mimic Gem5 simulation results and predict instructions per cycle (IPC).
Gem5 is a popular computer architecture simulator, and this project aims to predict IPC values based on various statistics extracted from Gem5 simulations.

Features:
      Utilizes machine learning models, including Random Forest, Linear Regression, and Neural Networks, to predict IPC values.
      Generates a CSV file from Gem5's stats.txt output to use as input data for training and testing the models.
      Collects various statistics from Gem5, including:
                        numLoadInsts
                        numStoreInsts
                        numInsts
                        numBranches
                        intAluAccesses
                        numOps
                        model (other optional statistics can be added)
      
Evaluation:
      Evaluate the model's performance and analyze the predictions.

Model Performance:
      Provide information on the performance of your machine learning models, such as accuracy, mean squared error, or any other relevant metrics.





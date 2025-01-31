# Data Science Coding Challenge

This application predicts the number of Ski tickets sold in the following data range: 0.12.2022 - 15.04.2023. 

## Usage: 
```docker compose up```
```docker compose down```

## Answers to the Questions: 

a. What do you see (patterns/potential issues)?
I see that we have the data available for only 5 months each Ski season (approx. from mid December until mid April). The first 4 Ski seasons have continuous data, the last 2 seasons (2020-2021 and 2021-2022) contain clearly damaged or missing data, therefore I decided to not use them. I also attach the Jupyter notebook where I did the data analysis. 

b. Which forecasting model did you choose and why?
I chose Prophet since it's easy to use (=allows to quickly build a prototype)

c. Are there steps you were unable to finish?
Tests 
Model tuning
Additional data incorporation
d. Suggest 3 additional steps/features to add.
Please see the answer to the previous question. Additionally, technical performance tests should be made for the deployment (e.g., the responce time test, load test, etc.).
e. Visualize your results: Create graphs to support your findings. 
Please see the forecast_output folder. 
f. Validate your predictions: Implement a validation method. 
A very basic validation was implemented (predicted number > or < 0).
g. Dockerize your application: Make it easily deployable.
Done. 
e. How did you use AI to help you in the process?
I used it to help me quickly create a foundation to build upon. For example, I asked it to create the docker file and write a code for training the Prophet model. 
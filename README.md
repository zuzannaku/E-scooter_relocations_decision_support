## Micromobility relocation support system

Project reserach question: How can machine learning be used to predict prolonged e-scooter idling to support fleet relocation decisions?

Project objective: Develop a machine learning based decision support system for VOI (micromobility) that classifies shared e-scooters scooters for relocation based on idle time duration using historical data. The system will have a user interface that visualizes the findings and recomendation though an interactive dashboard.

Business problem: Fleet managemnt and scooter relocation is an expensive operation cost. If not done efficienly some scooters may remain idle for extended periods of time and stay located in ares with low demand. This could lead to an inefficient distribution of the fleet and make finding availiable scooters by users more difficult, ultimately resulting in customer dissatisfaction. At the same time, unnecessarily relocating scooters creates additinal operational costs.

Project description: The aim of this proejct is to predict if e-scooters will remain idle for a long time. This will be achieved using ML. Then to make the process of relocating easier I used clusteting to point to ares where there is multiple scooter candidates with high probability of remaining idle for more than 8 hours. This way the costly relocation can be more sctructured and efficent.

ML task and pipeline design:
- type: supervised classification
- target value: "needs_relocation"
- models: decision tree, random forest, XGBoost
- evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

Data source: 
- scooters: https://zenodo.org/records/16947276
- weather API: https://open-meteo.com/en/docs/historical-weather-api

  
(repository does not include the raw idling and trip datasets because if their too big size)

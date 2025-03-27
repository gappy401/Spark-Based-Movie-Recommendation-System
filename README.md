# Movie Recommendation System

This repository contains the implementation of a robust **Hybrid Movie Recommendation System**, leveraging **PySpark** for distributed data processing and **FastAPI** for serving predictions through APIs. The project dynamically combines **Alternating Least Squares (ALS)** and **Linear Regression (LR)** models to offer accurate and personalized movie recommendations.

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contributing](#contributing)

## Overview
The system predicts movie ratings by dynamically choosing or hybridizing ALS and LR models based on user and movie metadata availability. This approach ensures:
1. **Personalization** using ALS for returning users.
2. **Metadata-driven predictions** via LR for new users or incomplete datasets.
3. A hybrid model that optimally combines predictions from ALS and LR for enhanced accuracy.

---

## Folder Structure

├── pycache/                    # Compiled Python files ├── data/                           # Raw and preprocessed datasets │   ├── raw_data.parquet            # Original data files │   ├── preprocessed_data.parquet   # Cleaned and feature-engineered dataset ├── models/                         # Trained machine learning models │   ├── als/                        # Saved ALS model │   ├── lr/                         # Saved Linear Regression model ├── notebooks/                      # Jupyter Notebooks for data analysis and modeling │   ├── Data_Pulling_Preprocessing_v8_0.ipynb  # Data preprocessing and feature engineering │   ├── Hybrid_Recommender_EDA_Modelling.ipynb # Exploratory data analysis and hybrid model development ├── app.py                          # FastAPI implementation with endpoints ├── User-Movie-data.parquet         # Main dataset for training/prediction ├── README.md                       # Documentation (this file) ├── requirements.txt                # Python dependencies for the project
---

## Features
- **Hybrid Recommendations**: Combines ALS and LR models with adjustable weighting for better predictions.
- **Dynamic Model Selection**: Automatically selects ALS or LR based on data availability.
- **FastAPI Integration**: Exposes endpoints for seamless interaction and real-time predictions.
- **PySpark for Scalability**: Handles large-scale data processing efficiently.
- **Easy Deployment**: Model saving/loading and API structure simplify deployment.

---

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/movie-recommendation-system.git
    cd movie-recommendation-system
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure PySpark and FastAPI are installed and configured.

---

## Usage
1. **Run the FastAPI Application**:
    ```bash
    uvicorn app:app --reload
    ```
2. **Access API Documentation**:
   Navigate to `http://127.0.0.1:8000/docs` for interactive API documentation using Swagger UI.

3. **Test Predictions**:
   Use the `/predict/` endpoint to get movie recommendations.

---

## API Endpoints
- **POST `/predict/`**:
    - **Description**: Predict the rating for a user and movie combination.
    - **Input**:
        ```json
        {
            "UserID": 123,
            "MovieID": 456,
            "Gender": "M",
            "Age": 25,
            "Occupation": "Engineer",
            "Genres": "Action|Comedy",
            "Year": 2020,
            "Runtime": 120,
            "IMDBRating": 8.5
        }
        ```
    - **Output**:
        ```json
        {
            "UserID": 123,
            "MovieID": 456,
            "PredictedRating": 4.5
        }
        ```

---

## Future Improvements
- Integrate additional metadata features (e.g., director, cast).
- Explore deep learning approaches for hybridization.
- Add a recommendation ranking endpoint.
- Optimize RMSE evaluation using parallelization.
- Enhance logging and error handling for production-grade deployment.

---

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

---

Happy Coding! 🎬✨

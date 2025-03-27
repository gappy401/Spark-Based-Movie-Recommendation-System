
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

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

---

Happy Coding! 🎬✨

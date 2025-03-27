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

## Folder Structure

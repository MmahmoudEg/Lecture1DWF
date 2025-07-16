# Amazon Recommender System

## Overview
This project builds a recommender system for an Amazon sales dataset using three approaches:
- **SVD (Baseline)**: Collaborative filtering with matrix factorization.
- **kNN**: User-based collaborative filtering with cosine/Pearson similarity.
- **Autoencoder**: Neural network with two hidden layers.

The dataset is preprocessed to handle mixed types, split multi-user reviews, and create a sparse user-item matrix. Models are evaluated with 5-fold cross-validation on RMSE, Precision@10, and Recall@10.

## Dataset
- **Source**: Scraped Amazon sales data (`amazon2.csv`), available at [link if public, e.g., Kaggle].
- **Size**: 2,587,373 rows, 7,705 unique products.
- **Preprocessing**:
  - Converted `rating` and `rating_count` to numeric types.
  - Split comma-separated `user_id` into individual rows.
  - Added sentiment scores using VADER sentiment analysis.
  - Created a sparse user-item matrix for efficiency.
  - Handled missing values and duplicates.

## Experimental Setup
- **Models**:
  - SVD: Surprise library, tuned for `n_factors`, `lr_all`, `reg_all`.
  - kNN: User-based, tuned for `k` and similarity metric.
  - Autoencoder: Keras, tuned for hidden units and dropout rate.
- **Cross-Validation**: 5-fold CV for HPO.
- **Hyperparameters**:
  - SVD: `n_factors` ∈ [50, 100, 200], `lr_all` ∈ [0.005, 0.01], `reg_all` ∈ [0.02, 0.1].
  - kNN: `k` ∈ [10, 20, 40], `sim_options` ∈ ['cosine', 'pearson'].
  - Autoencoder: Hidden units ∈ [50, 100], dropout rate ∈ [0.2, 0.5], epochs = 10.
- **Metrics**: RMSE, Precision@10, Recall@10.
- **Hardware**: Local machine (16GB RAM, Intel i7 CPU) or Google Colab with GPU for Autoencoder.

## Key Findings
- **SVD**: Best RMSE = 0.2593, Precision@10 = 0.9856, Recall@10 = 0.9988 (example values).
- **kNN**: [Computed dynamically].
- **Autoencoder**: Best RMSE = 1.8902, Precision@10 = 0.9941, Recall@10 = 0.7819 (example values).
- SVD outperforms Autoencoder in RMSE and Recall@10, while Autoencoder has slightly higher Precision@10 (1.3% better).
- kNN provides a simpler alternative but may struggle with scalability.
- **Challenges**:
  - High sparsity (~99% zeros in user-item matrix) impacts Autoencoder performance.
  - Cold-start problem for new users/products, especially for SVD and kNN.
  - Autoencoder requires significant computational resources.

## Challenges and Literature Context
- **Sparsity**: Common in recommender systems ([Xiang et al., 2010]). Hybrid models like NeuMF could improve performance.
- **Cold-Start**: SVD and kNN struggle with new users ([Koren et al., 2009]). Autoencoders can integrate side features (e.g., sentiment scores).
- **Scalability**: Autoencoders are compute-intensive ([He et al., 2017]).

## How to Navigate and Rerun Experiments
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
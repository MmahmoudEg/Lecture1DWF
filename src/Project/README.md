# Amazon Sales Dataset Recommendation System using Collaborative Filtering and Autoencoders

## Overview

This project aims to build a robust recommender system for the Amazon Sales Dataset. We explore and compare two distinct approaches: **Collaborative Filtering (CF)**, specifically using Singular Value Decomposition (SVD) and K-Nearest Neighbors (KNN), and **Autoencoders (Neural Networks)**. The primary goal is to predict user ratings and recommend top products based on these predictions.

## Group Members
* **Tasneem Shaheen** (ID: 107279)
* **Mostafa Khalid** (ID: 106699)
* **Medhansh Ahuja** (ID: 105982)

## Project Goal

The core objective is to develop a recommender system capable of:
1.  Predicting user ratings for unrated products.
2.  Generating personalized top product recommendations for users.
3.  Comparing the performance of Collaborative Filtering (SVD, KNN) and Autoencoder models in this context.

## Dataset
The project utilizes an Amazon Sales Dataset. This dataset contains information about various products, user interactions (ratings), and product details.

**Key Dataset Information:**
* **Shape:** (1465, 16)
* **Unique Products:** 1351
* **Columns include:** `product_id`, `product_name`, `category`, `rating`, `user_id`, `review_content`, etc.
## Data Processing and Preprocessing Steps
1.  **Load Data:** The `amazon.csv` dataset is loaded using pandas.
2.  **Initial Inspection:** Displaying dataset shape, unique product IDs, and basic info (`df.info()`, `df.describe()`).
3.  **Missing Value & Duplicate Check:** Verified no critical missing values or duplicate rows in `user_id`, `product_id`, and `rating`.
4.  **Rating Conversion:** The `rating` column is converted to a numeric type.
5.  **Data Subset:** Creating a `data` DataFrame with only `user_id`, `product_id`, and `rating`.
6.  **ID Encoding:** `user_id` and `product_id` are transformed from text to numeric IDs using `LabelEncoder` for model compatibility.
7.  **Pivot Table Creation:** A user-item interaction matrix (`pivot_table`) is created, with missing values filled with 0.

## Experimental Setup and Key Findings

Our experimental setup involved training and evaluating three different recommender models: SVD, KNN, and an Autoencoder. We split the preprocessed data into training and testing sets (80/20 split) to evaluate model performance on unseen data.

### Approaches Implemented
#### 1. Collaborative Filtering (CF)
Collaborative Filtering is a technique that makes automatic predictions (filtering) about a user's interests by collecting preferences or taste information from many users (collaborating).

* **SVD (Singular Value Decomposition) Matrix Factorization:**
    * **Setup:** The SVD model was trained using the `surprise` library with `n_factors=50` and `n_epochs=20`.
    * **Performance (SVD):**
        * RMSE: 0.2593
        * Precision@10: 0.9856
        * Recall@10: 0.9988

* **KNN (K-Nearest Neighbors) Collaborative Filtering:**
    * **Setup:** The KNN model (specifically `KNNBasic` from `surprise`) was chosen. Cross-validation was performed to find the best `k` and similarity metric (`pearson` and `cosine`) for user-based recommendations. The best performing parameters were `k=10` with `pearson` similarity.
    * **Performance (kNN - best model):**
        * RMSE: 0.2574
        * Precision@10: 0.9856
        * Recall@10: 1.0000

#### 2. Autoencoders (Neural Network Model)
Autoencoders are a type of neural network used for unsupervised learning of efficient data codings (features) in an unsupervised manner. In recommender systems, they learn latent representations of user preferences and reconstruct user-item interaction patterns.

* **Model Architecture:**
    * **Input Layer:** `(None, 1350)` (representing item ratings)
    * **Encoder Layers:** Dense layers with ReLU activation (`256`, `128`, `64` units)
    * **Decoder Layers:** Dense layers with ReLU activation (`128`, `256` units)
    * **Output Layer:** Dense layer with linear activation (`1350` units)
    * **Total Trainable Parameters:** 775,302
    * **Loss Function:** A custom `masked_mse` loss function was implemented to ignore zero (unrated) values during training, focusing only on actual observed ratings.
    * **Training:** The model was compiled with the Adam optimizer (learning rate=0.001) and trained for 50 epochs with a batch size of 32, using a 20% validation split.

* **Performance (Autoencoder):**
    * RMSE (non-zero ratings only): 1.8938
    * Precision@10: 0.9966
    * Recall@10: 0.7715

### Key Findings and Model Comparison
| Aspect | SVD Matrix Factorization | Autoencoder Neural Model | kNN Collaborative Filtering |
| :------------------- | :----------------------------------- | :---------------------------------------- | :--------------------------------------- |
| **Modeling style** | Linear latent-factor model | Nonlinear deep model | Memory-based, neighborhood model |
| **Data representation** | Sparse user–item matrix | Same matrix, but encoded through layers | Sparse user-item matrix |
| **Interpretability** | Higher (latent factors interpretable) | Lower (deep layers are a black box) | Moderate (based on similar users/items) |
| **Training speed** | Fast (few parameters) | Slower (many weights, backpropagation) | Can be slow for large datasets (similarity matrix computation) |
| **Scalability** | Scales well with large sparse data | May need more compute and memory | Scales less efficiently with dense data |
| **Cold-start** | Struggles with it | Can integrate side features more naturally | Struggles with it |
| **Typical use case** | Quick baseline with solid RMSE | Advanced scenarios needing richer signals | Simple, effective baseline for sparse data |
| **Sparsity Handling** | Handles sparse data well | Struggles with sparsity (often needs dense inputs or masked loss) | Handles sparse data well |

**Detailed Analysis of Results:**
* **RMSE:** SVD (0.2593) and kNN (0.2574) models demonstrated significantly lower RMSE compared to the Autoencoder (1.8938). This indicates that the traditional collaborative filtering methods were more accurate at predicting known ratings on this specific dataset.
* **Precision@10:** The Autoencoder achieved a slightly higher Precision@10 (0.9966) than SVD and kNN (0.9856). This suggests that for the top 10 recommendations, the Autoencoder had a marginally better ratio of relevant items.
* **Recall@10:** SVD (0.9988) and kNN (1.0000) showed substantially higher Recall@10 than the Autoencoder (0.7715). This implies that SVD and kNN were more effective at retrieving a larger proportion of all truly relevant items for a user within their top 10 recommendations.
**Overall Conclusion:**
Based on the RMSE and Recall@10, **SVD and kNN collaborative filtering models generally performed better** for this Amazon Sales Dataset. The Autoencoder, while showing promising precision, struggled with overall prediction accuracy and comprehensive recall, likely due to the inherent sparsity of the user-item interaction matrix.

## Challenges and Solutions in Recommender Systems

Building effective recommender systems comes with several well-known challenges, which are actively researched in the ML literature. Our project encountered some of these:

1.  **Data Sparsity:**
    * **Challenge:** The user-item interaction matrix is often very sparse, meaning most users have only rated a small fraction of available items. This makes it difficult for models to learn meaningful patterns. Our dataset had a sparsity of **99.91%**.
    * **Existing Solutions/Literature:** Matrix factorization methods (like SVD) are inherently designed to handle sparsity by finding latent factors. For neural networks (like Autoencoders), common solutions include using masked loss functions (as implemented with `masked_mse` to ignore zero/unrated entries), or incorporating side information.
    * **Our Approach:** We explicitly used `fillna(0)` for the pivot table, and for the Autoencoder, we implemented a `masked_mse` loss to ensure the model only learned from observed ratings, which is a standard practice to mitigate sparsity issues in neural collaborative filtering. Despite this, the Autoencoder's performance on RMSE suggests sparsity still posed a challenge for this model.

2.  **Cold-Start Problem:**
    * **Challenge:** Recommending items to new users or recommending new items that have few or no ratings. Collaborative filtering models struggle here because they rely on historical interactions.
    * **Existing Solutions/Literature:** Content-based methods, hybrid approaches, and incorporating side features (e.g., user demographics, item attributes) are common solutions. Neural network models (like Autoencoders) can sometimes integrate side features more naturally.
    * **Our Approach:** While our current implementation primarily focuses on collaborative filtering based on existing ratings, the Autoencoder's architecture is more amenable to future extensions that could integrate content-based features to address cold-start scenarios. This is highlighted in our "Future Work."

3.  **Scalability:**
    * **Challenge:** As the number of users and items grows, training and generating recommendations can become computationally expensive.
    * **Existing Solutions/Literature:** Distributed computing frameworks (e.g., Spark for SVD), approximate nearest neighbor algorithms for KNN, and more efficient neural network architectures are used.
    * **Our Approach:** For this dataset size, standard `surprise` and `tensorflow` implementations were sufficient. However, for larger datasets, distributed solutions would be necessary.

4.  **Interpretability:**
    * **Challenge:** Understanding *why* a recommender system makes a particular recommendation, especially with complex deep learning models, can be difficult.
    * **Existing Solutions/Literature:** SVD's latent factors offer some interpretability. Attention mechanisms in neural networks or post-hoc explanation techniques are being researched for deep learning models.
    * **Our Approach:** SVD offers higher interpretability in our findings, while the Autoencoder remains more of a "black box."

## Repository Navigation

The project is structured as follows:

* `Recommender_System.ipynb`: The main Jupyter Notebook containing all the code for data processing, model training, evaluation, and comparison.
* `amazon.csv`: The dataset used for this project. (Please ensure this file is present in the root directory).
* `requirements.txt`: That have all the libraries that need to be installed.
* `README.md`: This file.

## How to Rerun Experiments
To reproduce the results and run the experiments:
1.  **Clone the repository:**
```bash
    git clone https://github.com/MmahmoudEg/Lecture1DWF.git
    cd src/Project
```

2.  **Install dependencies:**
    It's highly recommended to use a virtual environment to manage project dependencies.
```bash
    # Create a virtual environment
    python -m venv venv
    # Activate the virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate

    # Install required packages
    pip install pandas scikit-learn matplotlib seaborn numpy surprise tensorflow
```
    *(Note: `surprise` is a specific library for recommender systems. `tensorflow` is for the Autoencoder.)*

3.  **Download the dataset:**
    Ensure you have the `amazon.csv` file in the root directory of your cloned repository. If it's not included in the repository, you will need to download it separately and place it there.
4.  **Open and run the Jupyter Notebook:**
    ```bash
    jupyter notebook Recommender_System.ipynb
    ```
    This command will open the Jupyter Notebook in your web browser. You can then run all cells sequentially from top to bottom to execute the entire analysis, model training, and evaluation.

## Future Work
* **Hyperparameter Tuning:** Further optimize model performance by more extensive hyperparameter tuning for all models (SVD, KNN, and Autoencoder).
* **Hybrid Models:** Explore combining collaborative filtering and autoencoder approaches for potentially better performance.
* **Content-Based Features:** Incorporate product features (e.g., `about_product`, `category`) as side information, especially for the Autoencoder, to improve cold-start recommendations.
* **Scalability:** Investigate methods for handling larger datasets, such as distributed computing frameworks.
* **Deep Dive into Autoencoder Loss:** Analyze the high `val_loss` in the Autoencoder training history and explore regularization techniques or different architectures to address it.
---

**Thank you for exploring this project!**
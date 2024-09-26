# Online-News-Popularity
This project focuses on predicting the popularity of articles published by Mashable based on various attributes related to the article's content and metadata. The dataset used contains 61 attributes, including statistics like word counts, number of images and videos, sentiment polarity, and keyword information. The goal is to either predict the number of shares (regression) or classify the articles into low or high popularity (classification), based on whether the number of shares is below or above 1,400.

Task Descriptions:
        -Regression: Predict the exact number of shares an article will receive using the provided           attributes. The model's performance is evaluated using Root Mean Squared Error (RMSE).
        -Classification: Classify articles into two categories:
            -Low popularity: shares < 1,400
            -High popularity: shares ≥ 1,400 The classification model is evaluated using accuracy,               which measures the proportion of correct predictions.

Models and Approaches:
As part of this project, I explored several machine learning algorithms, both for regression and classification tasks, including:

        -Decision Trees
        -Linear Regression and Linear Classification
        -Classification as Regression (logistic regression)
        -K-Nearest Neighbors (KNN) for both classification and regression tasks
        -Neural Networks for both regression and classification tasks

This project represents my first approach to the field of Artificial Intelligence and Machine Learning. Through hands-on experimentation with different algorithms, I gained a deeper understanding of the strengths and limitations of each approach, particularly in dealing with real-world datasets like this one. I learned to apply decision trees, linear models, KNN, and neural networks to various tasks and to assess their performance using the appropriate metrics for each problem type.

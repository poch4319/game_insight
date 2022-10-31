Game Review Insight Ectraction Using Sentiment Analysis
=========================

This repo contains my approach in extracting the insight from game reviews so that game developer can make use of the insight to develope their games better.

In this repo, I developed several sentiment alaysis model trained using game review dataset, so that they are able to recognize negative or positive reviews. Once the models are trained, I used the best performing model to label new data, and collected the data that are labeled as negative reviews for key-word anaylsis. Based on the keywords found in the negative data, we can provide insight on how a games can be improved.

A potential use case of our model is to collect textual game reviews regarding to a specific games, have the review sentiment analysis model differentiate the positive and negative reviews, then perform keyword analysis on the labeled reviews to figure out what do people talk about mostly in positive reviews or negative reviews. With the insight collected, game developer can understand their customers better and modify their games accordingly.

**The dataset**: The dataset of amazon game reviews can be found here [this dataset website](https://nijianmo.github.io/amazon/index.html). Each data contains textual review or comment about the game, and how many stars (1-5) the reviewer gave to the game. Note that the dataset do not contain any sentiment label (positive or negative) that can be directly used as the target label for training our sentiment analysis model, so the preprocessing and relabelling are needed.

## Preprocessing, Relabelling, Dictionary-Based Model
This part can be found with detailed description in `EDA_dict_base_model.ipynb`. Essentially I have to resample the dataset because it was very imbalanced with overwhelmingly high amount of 5 stars review. Then I relabel the data with 1 star as `negative`, 2-4 stars as `neutral` and 5 star as `positive` for training my model. In this notebook I also explore a `dictionary-based` sentiment analysis model, which is a very simple algorithm that rates a comment based on how many words in it are found in a dictionary that represent negative words in English and how many are found in the positive.

## Bert-Based Sentiment Analysis Model and Keyword Analysis
In `neural.ipynb` I compared the performance of pretrained BERT sentiment analysis model and a version where I fine-tuned with my game review data. Packages involved `huggingface transformer` and `simpletransformers` for easy finetuning. As predicted, fine-tuned version of `BERT` yields the best result. I then used the best version of model to label extra game data and perform the keyword analysis using `keyness` package to get the keyword on data labelled with negative as well as the ones labelled with positive. The result indicate the insight in both negative comments and positive comments.
```

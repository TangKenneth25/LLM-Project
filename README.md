# Large Language Model Project for Lighthouse Labs

## Project Task
Create a sentiment analysis tool that uses an LLM to interpret emotions in text data from social media platforms. Various models including pretrained models and tranfsfer learning models were generated with their performance compared using F1-score.

## Dataset
I am using the popular stanfordnlp imbd movie review dataset. This dataset contains 25,000 movie reviews for training, 25,000 for testing, and 50,000 additional unlabelled rows. 

[Link to dataset on huggingface.co](https://huggingface.co/datasets/stanfordnlp/imdb)


## Base model
For the base model, a BoW (bag of words) tokenization sklearns logistic regression was used. BoW is a simple tokenization technique that takes each word as a token. Logistic Regression was chosen for its simplicity and efficiency, which makes it a popular baseline model for comparing with more complicated models.

## Pre-trained Model
For the pre-trained model, [sentiment-roberta from siebert](https://huggingface.co/siebert/sentiment-roberta-large-english) was used. There was a max token limit in this model, and the data was truncated to 200 words to ensure that all data was within the limit after the pre-trained tokenizer. This pretrained model is a popular sentiment prediction model that was selected due to the more robust NLP with the same classification outputs as this project (positive or negative).

Following that, a fine-tuned model using [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) was trained. DistilBERT was selected as it is beneficial to have a smaller, lighter model would allow for faster training of the model for the purposes of this project. Further hyperparameter tuning was attempted but the increase in performance following the initial fine-tuning was insignificant.

The final fine-tuned model was uploaded to my models on huggingface and can be viewed here: [https://huggingface.co/ktang5/imdbSentimentAnalysis](https://huggingface.co/ktang5/imdbSentimentAnalysis)

## Performance Metrics
F1-score was used as the primary performance metric for this project as it works well with binary classification and accounts for both precision and recall. 

Below is a summary of the results of the models used in this project

| Model             | Model Name                                    | F1-Score  |
|----------         |----------                                     |---------- |
| Base Model        | Bag of Words - Logistic Regression            | 0.87      |
| Pretrained Model  | sentiment-roberta-large-english by Siebert    | 0.90      |
| Fine-Tuned Model  | distilbert-base-uncased                       | 0.93     |
| Optimized Model   | distilbert-base-uncased                       | 0.93      |

We can see that the base model performed quite well, with some improvements using the RoBERTa pretrained model. We can see slightly further improvements in the fine-tuned DistilBERT model. 

## Hyperparameters
Hyperparamter tuning was attempted with a sample of the overall data so that trials with different parameters could be ran quickly. The learning rate, weight decay, warmup steps, and epochs were adjusted throughout the differrent trials. 

However, we started with a model with parameters that produced an F1-score of 0.929, so it was already a very good model. Adjusting these parameters did not really produce any significant improvement to the model.

These were the training arguments that were trialed, all of which resulted in a similar F1-score as the original fine-tuned model

| Parameter             | Value                         |
|----------             |----------                     |
| learning_rate         | 1e-4, 1e-5, 2e-5, 3e-5, 5e-5  |
| num_train_epochs      | 1, 2, 3                       |
| weight_decay          | 0.1, 0.01, 0.005, 0.001       |
| warmup_steps          | 2000                          |
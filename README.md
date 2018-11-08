# Twitter_sentiment_analysis

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

For finding the accuracy of different model in classification of tweets I am using *F1 Score*.

The F1 Score is the *2*((precision*recall)/(precision+recall))*. It is also called the F Score or the F Measure.

# Step taken to classify:

>*Twitter preprocessing ang cleaning*  

The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

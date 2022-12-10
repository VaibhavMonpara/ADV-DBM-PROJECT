import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Uber Reviews").getOrCreate()


# ---------------------------------------------------------------------------------------------------------
# Data Loading
df_original = spark.read.csv(
    "/Users/csuftitan/Documents/Semester 1/Advanced DBMS/ADV-DBM-PROJECT/Uber_Ride_Reviews.csv", header=True,
    inferSchema=True)
df_original.show()


# ---------------------------------------------------------------------------------------------------------
# Data cleaning

# removing (3) rating records from the dataset
df_clean = df_original.filter(df_original.ride_rating != 3)

# ---------------------------------------------------------------------------------------------------------
# tokenization
tokenizer = Tokenizer(inputCol="ride_review", outputCol="ride_review_tokens")
df_clean = tokenizer.transform(df_clean)

# ---------------------------------------------------------------------------------------------------------
# remove custom stop words
stop_words = ['ourselves', 'hers', 'between', 'yourself', 'driver', 'uber', 'again', 'there',
              'about', 'once', 'during', 'very', 'having', 'with', 'they', 'own', 'an', 'be',
              'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other',
              'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves',
              'until', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
              'were', 'her', 'more', 'himself', 'this', 'should', 'our', 'their', 'while',
              'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before',
              'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
              'then', 'that', 'because', 'what', 'why', 'so', 'can', 'now', 'he', 'you', 'herself',
              'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'whom',
              'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here']

stop_words_remover = StopWordsRemover(inputCol='ride_review_tokens', outputCol='ride_review_clean',
                                      stopWords=stop_words)
df_clean = stop_words_remover.transform(df_clean)
df_clean.show()

# ---------------------------------------------------------------------------------------------------------
# converting list of strings of a review to list of integers based on
# frequency of each word in the corpus(total reviews)
cv = CountVectorizer(inputCol="ride_review_clean", outputCol="features")
cv_model = cv.fit(df_clean)
cv_train_df = cv_model.transform(df_clean)
cv_train_df.show()

# split train and test data
train, test = cv_train_df.randomSplit([0.7, 0.3])

# ---------------------------------------------------------------------------------------------------------
# train Logistic Regression model
lr = LogisticRegression(featuresCol='features', labelCol='sentiment')
lrModel = lr.fit(train)


# ---------------------------------------------------------------------------------------------------------
# Plotting the regression graph
beta = np.sort(lrModel.coefficients)

plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

# ---------------------------------------------------------------------------------------------------------
# prediction
predict_test = lrModel.transform(test)
predict_test.select("ride_review_clean", 'sentiment', 'prediction', 'ride_rating').show()

# Words and their respective coef
df = pd.DataFrame({'Word': cv_model.vocabulary, 'Coef': lrModel.coefficients})
df.sort_values('Coef').head(20)

# Inspect tool
predict_test.filter(predict_test['prediction'] == 1) \
    .select("sentiment", "ride_rating", "probability", "prediction", "ride_review") \
    .orderBy("probability", ascending=False) \
    .show(n=50, truncate=50)

labels_and_scores = predict_test.select('sentiment', 'prediction')
labels_and_weights = labels_and_scores.collect()
labels_and_weights.sort(key=lambda x: x[1], reverse=True)
labels_by_weight = np.array([k for (k, v) in labels_and_weights])

length = labels_by_weight.size
true_positives = labels_by_weight.cumsum()
num_positive = true_positives[-1]
false_positives = np.arange(1.0, length + 1, 1.) - true_positives

true_positive_rate = true_positives / num_positive
false_positive_rate = false_positives / (length - num_positive)


# ---------------------------------------------------------------------------------------------------------
# Generate layout and plot data
fig, ax = plt.subplots(figsize=(10.5, 6), facecolor='white', edgecolor='white')
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(false_positive_rate, true_positive_rate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model

# Evaluate model

# evaluation of the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="sentiment")
predict_test.select("sentiment", "prediction", "probability").show(truncate=False)

# Topic_Modelling_Crypto
The aim of this project is to discover the abstract topics that occur in the collection of data from Reddit posts and comments  of different users based on the discussions on various Cryptocurrency. This proposed approach will be able to draw out and  classify the posts and comments to different topics and cluster the frequent similar information together. 

We will also be seeing how the sentiment of the daily discussions/posts impact the change in the bitcoin prices by checking the 
correlation between them. 

This is a quick guide on installation of the libraries required to run the codes in the below mentioned sequence.  
The project consists of these files-
-Rudraksh Mishra- Dominic Thomas - Nikhitha Kunduru-Data Exctraction-CODE.ipynb
-Rudraksh Mishra- Dominic Thomas - Nikhitha Kunduru- Data Pre-Processing and Sentiment Analysis -CODE
-Rudraksh Mishra- Dominic Thomas - Nikhitha Kunduru-Data Exploration Analysis-CODE.ipynb
-Rudraksh Mishra- Dominic Thomas - Nikhitha Kunduru-Data Modelling-CODE.ipynb
-Rudraksh Mishra- Dominic Thomas - Nikhitha Kunduru-Correlation-CODE.ipynb

Data Exctraction file:
1) To run the Data Exctraction model we need to install and import these libraries:
- pandas
- datetime
- praw
- psaw
- pmaw
2) Running the code:
After the installation of the required libraries. The code can be run after specifying the time frame for which the posts have to be scraped and specifying the location to load the csv file.
Then the csv file is uploaded in the specified location with the data. 

Data PreProcessing and Sentiment Analysis file:
Note: On trying to run PreProcessing in Jupyter Notebook, we were getting memory out errors. Hence we have submited it as a python script.
1) To run the Data PreProcessing we need to install and import these libraries:
- pandas
- datetime
- numpy
- NLTK
- textblob
- afinn
- dateutil

2) Running the code:
After the installation of the required libraries. The code can be run after inputting the location where the files extracted from Data Exctraction step are stored. Please take care to ensure that the specific folder contains only the input files
Then with the change of ouput path you can run the code.
The code will run successfully and save the to the location specified in the output file. Please change to appropriate location 

Data Exploration Analysis file:
1) To run the Data Exctraction model we need to install and import these libraries:
- pandas
- matplotlib
- seaborn
- wordcloud
2) Running the code:
After the installation of the required libraries. The code can be run after inputting the processed sentiment analysis dataset location.

Data Modelling file:
1)To run the topic models we need to install and import these libraries:
- Pandas
- Numpy
- Nltk
- Re
- String
- Spacy
- Gensim
- PyLDAvis
- Scipy
- Sklearn
- Pickle
- Collections
- Bokeh
- Matplotlib
- Seaborn
2)Running the code:
After the installation of the required libraries. The code can be run after inputting the cleaned and prepocessed dataset location.
The hyperparameter tuning and number of topics optimizer might take sometime to run. 

Correlation file:
1)To run the topic models we need to install and import these libraries:
- Pandas
2)Running the code:
After the installation of the required libraries. The code can be run after inputting the cleaned and prepocessed dataset location.


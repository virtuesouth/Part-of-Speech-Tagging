# Part-of-Speech-Tagging

Hello everyone. This is my Python code for my second assignment in natural language processing. The prerequisites for the assignment were as follows. For this assignment, the professor shared with us a Brown Corpus with separated test and train sets. Therefore, instead of using the 'os' library, I did a read operation on the file names directly. You can access the Brown corpus via kaggle https://www.kaggle.com/nltkdata/brown-corpus?select=brown.csv

In this assignment, you will create an HMM PartofSpeechTagger using the scaled down BrownCorpus.

DataSet (Brown_hw.rar)
DataSet is divided into two directories as Train and Test. There are 80 files in the Train directory and 8 files in the Test directory. Each row of the log holds a sentence (there are blank lines; you will not consider blank lines). An example sentence is as follows.

Attorneys/nns for/in the/at mayor/nn said/vbd that/cs an/at
amicable/jj property/nn settlement/nn has/hvz been/ben
agreed/vbn upon/rb ./.

It is separated from the Part-Of-Speech, which is marked with each word / character.

Your program should do the following:

Creation of First-Order-HMM for POS-Tagging:

- Your program will create the First-Order-HMM by reading all the files (80) in the Train directory. This means that all probability data required for the HMM will be collected from the Train set. In addition to the information required for HMM, you may need to collect other information (word frequencies, total word count,..).

- First convert all words to lowercase letters. So all words in your model will only consist of lowercase letters (and other characters).

- While creating the HMM model, assume that lowest 10 frequency words from the Train set as UNK (unknown) words. So you will pretend that the word UNK is used instead of these words. While finding the POS of the test set with the created HMM, if the word is not in the Train set, you will use the UNK word for that word.

Writing HMM Data Collected From Train Set to Logs:

- PosTags.txt: You will write the POStags in the train cluster together with their frequencies in the PosTags.txt file in descending order according to their frequencies (how     many times they passed in the test set). Each row of this log must contain a POS tag and its frequency. The first line should keep the most frequently passing POStag and its     frequency, and the last line the least passing POStag and its frequency. Each row of this log should look like the following.
  tag tag_frequency

- TransitionProbs.txt: You will write the TagTransitionProbability values you created for HMM into this file. Each row of this file should hold the following data.
  taga tagb P(tagb | taga)
  The lines of this file should be sorted first by taga and then alphabetically by tagb.

- Vocabulary.txt: Write all the words in the train set in descending order according to their frequencies, along with their frequencies and the most probable POSTag (Most Likely   Tag). Each row of this log should look like the following. 
  Word - Frequency of the word - MostLikelyTag
  In the first line of this file, print the total number of words and the number of individual words (Vocubulary Size) in the Train set.

- EmissionProbs.txt: You will write the EmissionProbability values of POSTags to this file. Each row of this log should look like the following.
  tag - word -  P(word | tag)
  These log lines should be sorted first by tag and then alphabetically by word.
  
- InitialProbs.txt: Write the probability that POSTags appear at the beginning of the sentence in this file. Each row of this log should look like the following.
  tag - P(tag | s )
  The lines of this file should be sorted alphabetically according to tag.
  
  Finding POSTags of the Test Set using HMM and writing the obtained results to a file.
  
- You will find the POSTags of the words in the file in the test set using the HMM you have created and calculate the success rates and write them in the result file. You must     first convert the words to lowercase.
  
- You will find the most probable POSTag order of each sentence (lines of file) in the test set with the help of the HMM and Viterbi algorithm you created. So you will find       the POSTag of every word in the sentence. You can find the success value of your model because it is in the correct PosTags in the test set.
  
- Result.txt: You will write your success results for the test set in this file. This file should contain the following information in order.

- Total number of words in the test set (all 8 logs)
- Number of words with correct POSTags in the test set (all 8 logs)
- Ca41 log results in test directory:
- the total number of words in the ca41 log
- Number of correct POSTag words found in ca41 file
- The sentences of the ca41 file must be written to the Result.txt file together with the found POSTags.

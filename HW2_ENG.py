import nltk
import collections
from glob import glob


# convert lowercase and delete whitespace
def clean(text):
    text = text.lower()
    text = " ".join(text.split())
    return text


# dictionary returning structure to keep frequency values
def dict_freq(liste):
    dict1 = {}
    for i in liste:
        if i in dict1:
            dict1[i] += 1
        else:
            dict1[i] = 1
    return dict1



def sorted_dict(sozluk):
    return sorted(sozluk.items(), key=lambda x: x[1], reverse=True)


# Reading files
raw_text = ''
filenames = glob('ca*') + glob('cb*') + glob('cc*')
test_files = ['ca41', 'ca42', 'ca43', 'ca44', 'cb26', 'cb27', 'cc16', 'cc17']  # test dosyaları

# Removing test files from the train list.
for i in test_files:
    if i in filenames:
        filenames.remove(i)


for i in filenames:
    ff = open(i)
    raw_text = raw_text + ff.read()


tagged_text = clean(raw_text)  # cleaning process

# scrapping words and tags
word_and_tag = [nltk.tag.str2tuple(t) for t in tagged_text.split()]
kelime_sayisi = len(word_and_tag)

# PosTags section
pos_counts = collections.Counter((subl[1] for subl in word_and_tag))

d1 = dict()  # Dictionary structure for finding tag frequencies.

f = open("PosTags.txt", "w")
f.write("tag - tag_frequency\n")
for i, j in pos_counts.most_common():
    d1[i] = j
    f.write(str(i) + "\t" + str(j) + "\n")  # Converted to str structure.
    # print(i, "-", j)

f.close()

# TransitionProbs
l1 = list()  # tag list to hold bigrams
for a, b in nltk.ngrams(word_and_tag, 2):
    l1.append((a[1], b[1]))

d2 = dict_freq(l1)
sorted_bigram = sorted_dict(d2)  # Sorting bigrams.

d3 = d2.copy()  # Dictionary structure for holding transition probabilities

for i, j in sorted_bigram:
    d3[i] = j / d1[i[0]]

TransitionProbs = sorted(d3.items(), key=lambda x: (x[0], x[1]),
                         reverse=False)  # sorted d3 prob. dictionary

f = open("TransitionProbs.txt", "w")
f.write("taga - tagb - P(tagb|taga)\n")
for i, j in TransitionProbs:
    f.write(str(i[0]) + "\t" + str(i[1]) + "\t" + str(j) + "\n")
# print(i[0], '-', i[1], '-', j)

f.close()

# Vocabulary

# Find word frequencies - to find unique word count and frequency before replacing them with unk
pos_counts = collections.Counter((subl[0] for subl in word_and_tag))

d4 = dict()  # Dictionary structure for finding word frequencies.
for i, j in pos_counts.most_common():
    d4[i] = j
unique_kelime_sayisi = len(d4.keys())  # unique word count before replacing with unk

#################       UNK       ################################## ---- Since the tags are not affected, they were made in the word section.

siralanmis_kelimeler = sorted_dict(d4)  # Sorting words by frequency

en_az_gecen = list()  # List structure to keep at least 10 last words
for i in range(len(siralanmis_kelimeler) - 1, len(siralanmis_kelimeler) - 11, -1):
    en_az_gecen.append(siralanmis_kelimeler[i][0])

"""frekansi_1_olan_kelimeler = list()   # List structure to keep words with frequency 1.
for i, j in d4.items():
    if j == 1:
        frekansi_1_olan_kelimeler.append(i)"""

dunk = list()  # To create word and tag structure with unk / cannot be changed because nltk works with tuple.
for i, j in word_and_tag:
    dunk.extend([[i, j]])

for i in range(0, len(dunk)):
    if dunk[i][0] in en_az_gecen:  # Adding unk to the 'word / tag' list for at least 10 last words.
        dunk[i][0] = 'unk'

word_and_tag_unk = list()  # for creating unk with word_and_tag
for i, j in dunk:
    word_and_tag_unk.extend([(i, j)])  # for avoid tuple

#################       UNK       ##################################

pos_counts = collections.Counter(
    (subl[0] for subl in word_and_tag_unk))  # post counts word count with variable word_and_tag_unk

d5 = dict()  # Dictionary structure for holding word frequencies after unk change.
for i, j in pos_counts.most_common():
    d5[i] = j

mostliketag = nltk.FreqDist(word_and_tag_unk)  # most likely tag function.

d6 = dict()  # Dictionary structure for holding the mostlike frequencies of words.

f = open("Vocabulary.txt", "w")
f.write("Total number of words" + " " + str(kelime_sayisi) + " " + "Vocubulary Size" + " " + str(
    unique_kelime_sayisi) + "\n")
f.write("word - Frequency of the word - MostLikelyTag\n")

for i, j in mostliketag.items():   # to move the most like tag of words to dictionary structure.
    d6[i[0]] = i[1]

for i, j in d5.items():
    f.write(str(i) + "\t" + str(j) + "\t" + str(d6[i]) + "\n")
    # print(i, '-', j , '-', d6[i])
f.close()

# EmissionProbs
d7 = dict_freq(word_and_tag_unk)  # Dictionary structure established to find the number of occurrences of words and tags.

# Sort the lines of the log in alphabetical order first by tag, then by word
sorted_word_tags = sorted(d7.items(), key=lambda x: (x[0][1], x[0][0]), reverse=False)

df_t = list()  # List structure holding word and tag probabilities to be used in test section
f = open("EmissionProbs.txt", "w")
f.write("tag - kelime - P(kelime|tag)\n")
for i, j in sorted_word_tags:
    f.write(str(i[1]) + "\t" + str(i[0]) + "\t" + str((j / d1[i[1]])) + "\n")
    # print(i[1], '-', i[0], '-', j/d1[i[1]] )
    df_t.extend([[i[0], i[1], (j / d1[i[1]])]])  # filling in the list part for the test.

f.close()

# InitialProbs

cumleler = nltk.sent_tokenize(tagged_text,
                              language='English')  # division into sentences. kept in the list structure.
cumle_sayisi = len(cumleler)

init = dict()  # Dictionary structure defined to find the tag frequency at the beginning of the sentence.
for i in range(0, cumle_sayisi):
    test = [nltk.tag.str2tuple(t) for t in cumleler[i].split()]  # separating the words of each sentence into tags.

    if test[0][1] in init:  # First element of the list is the tag of the first word of the sentence.
        init[test[0][1]] += 1
    else:
        init[test[0][1]] = 1

for i in init:
    init[i] = init[i] / cumle_sayisi  # To replace with new dictionary values by computing initial probabilities

sorted_init = sorted(init.items(), key=lambda x: x[0], reverse=False)  # Sorting by tag

f = open("InitialProbs.txt", "w")
f.write("tag - P(tag|<s>)\n")

for i, j in sorted_init:
    f.write(str(i) + "\t" + str(j) + "\n")
    # print(i, '-' ,j)
f.close()

ff.close()  # Closing train files.

########### SONUC BOLUMU ##########
f = open("Sonuc.txt", "w")
erdem = 1  # Control variable for printing the results for ca41.
while erdem <= 2:

    # Reading file
    raw_text_test = ''

    if erdem == 1:
        test_files = ['ca41', 'ca42', 'ca43', 'ca44', 'cb26', 'cb27', 'cc16', 'cc17']
    else:
        test_files = ['ca41']

    for i in test_files:
        fft = open(i)
        raw_text_test = raw_text_test + fft.read()

    tagged_text_test = clean(raw_text_test)  # cleaning

    # reading word and tag
    test_word_and_tag = [nltk.tag.str2tuple(t) for t in tagged_text_test.split()]
    test_kelime_sayisi = len(test_word_and_tag)

    if erdem == 1:
        f.write("Total number of words in the test set (all 8 logs)\t" + str(test_kelime_sayisi) + "\n")
    else:
        f.write("The total number of words in the ca41 log\t" +
                str(test_kelime_sayisi) + "\n")

    # reading sen.
    test_cumleler = nltk.sent_tokenize(tagged_text_test, language='English')

    kelime_tag_olasilik = list()  # List for keeping tags found according to HMM model.

# VITERBI ALGORITHM
    for i in range(0, len(test_cumleler)):
        test = [nltk.tag.str2tuple(t) for t in
                test_cumleler[i].split()]  # separating the words of each sentence into tags

        for k in range(0, len(test)):  # list to scan words within returned sentences.
            if test[k][0] in d5:  # If this word is not present in the test set, it is replaced with 'unk'.
                kelime = test[k][0]
            else:
                kelime = 'unk'

            if k == 0:  # The first index of this list is the first word and the initial probe must be calculated.
                maks = 0
                tag = 'boş'

                for z in range(0,
                               len(df_t)):  # Finding probability and tag of this first word from emission prob.
                    if kelime == df_t[z][0]:

                        if df_t[z][1] in init:  # Is the given tag included in the inital prob.?
                            init_prob = init[df_t[z][1]]  # Finding the possibility to tag through the init probe.
                            emis_prob = df_t[z][2]
                            olasilik = (emis_prob * init_prob)
                            if olasilik > maks:  # to find the maximum probability value
                                maks = olasilik
                                tag = df_t[z][1]

                        else:
                            init_prob = 0.00000000001  # If it is not included in inital prob, very small probability value is assigned.
                            emis_prob = df_t[z][2]
                            olasilik = (emis_prob * init_prob)
                            if olasilik > maks:
                                maks = olasilik
                                tag = df_t[z][1]

            else:
                onceki_olasilik = maks  # the previously found probability value.
                onceki_tag = tag  # the previous tag value.
                flag = 0
                maks = 0
                yeni_tag = 'boş'
                for z in range(0, len(df_t)):
                    if kelime == df_t[z][0]:
                        if (
                                onceki_tag,
                                df_t[z][1]) in d3:  # Checking if the bigram tag is included in the transition probe.
                            trans_prob = d3[(onceki_tag, df_t[z][1])]
                            emis_prob = df_t[z][2]
                            olasilik = (emis_prob * trans_prob * onceki_olasilik)
                            if olasilik > maks:
                                maks = olasilik
                                tag = df_t[z][1]
                        else:
                            trans_prob = 0.00000000001  # If it is not included, a very small value is assigned.
                            emis_prob = df_t[z][2]
                            olasilik = (emis_prob * trans_prob * onceki_olasilik)
                            if olasilik > maks:
                                maks = olasilik
                                tag = df_t[z][1]

            kelime_tag_olasilik.extend([(kelime, tag)])  # Word and probability values found with viterbi.
            onceki_tag = tag  # The previous tag becomes the new value found.
            onceki_olasilik = maks  # the previous probability value becomes the new value found.

    dogru_bulunan_tag_sayisi = 0
    for i in range(0, len(test_word_and_tag)):
        if kelime_tag_olasilik[i][1] == test_word_and_tag[i][1]:
            dogru_bulunan_tag_sayisi += 1

    if erdem == 1:
        f.write("Number of words with correct POSTags in the test set (all 8 logs)\t" +
                str(dogru_bulunan_tag_sayisi) + "\n")
    else:
        f.write("Number of correct POSTag words found in ca41 file\t" +
                str(dogru_bulunan_tag_sayisi) + "\n")

    if erdem != 1:  # Printing ca41 according to the words and tags found.
        for i in range(0, len(kelime_tag_olasilik)):
            f.write(str(kelime_tag_olasilik[i][0]) + "/" + str(kelime_tag_olasilik[i][1]) + " ")

    erdem += 1

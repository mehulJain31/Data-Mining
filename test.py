#Mehul Jain
# 1001229017

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
filename = './debate.txt'
file = open(filename, "r", encoding='UTF-8')
list=[]
doc = file.read() #read the file
list=doc.split('\n')
for i in list:
    if i=="": # remove extra spaces
        list.remove(i)
file.close()

#tokenize the document

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
temp=[]
paraNumber=1# for extracting para in query() function
paraDct={} # for extracting the paragraph in query() function
for i in list:
    paraDct.update({paraNumber:i})
    tokens = tokenizer.tokenize(i)# tokenize the doc line by line
    temp.append(tokens)
    paraNumber+=1

#stopwords: commonly used words. like a,an,the,or,in etc.
#stopword removal in debate.txt(doc)

stopwordList=set(stopwords.words('english'))

filteredList = []
for w in temp:
    word=[]
    for i in w:
        if i not in stopwordList: # removing stopwords
            word.append(i)
    filteredList.append(word)# append the non stopwords to the main tokenized list

#stem the document

stemmer = PorterStemmer()
count=0
for i in filteredList:
        count+=1
        for token in range(len(i)):
            i[token]=stemmer.stem(i[token])# stem every token


def getidf(token):
    N= len(filteredList) # get N: total no of documents for calculating ID-F (Numerator for ID-F)
    #idf calculation
    tokenDocCount=0 # no of documents containing token
    for i in filteredList:
        if token in i:
            tokenDocCount+=1 # count the frequency of the word in the document
    if tokenDocCount==0:
        return -1
    else:
        return math.log10(N/tokenDocCount) # return the tf values

print("%.4f" % getidf(stemmer.stem("immigration")))


def getqvec(qstring):
    N = len(filteredList)
    words = []
    # tokenizing,stemming and removing stopwords from the string
    tokenWord = tokenizer.tokenize(qstring)  # tokenize the string
    for i in tokenWord:
        if i.lower() not in stopwordList:
            words.append(stemmer.stem(i))  # stem and remove stopwords

    # tf calculation
    wordDict = {}
    totalWords = len(words)  # get total words for tf
    for i in words:
        wordDict.update({i: words.count(i)})  # dictionary with the words and their count

    for key, value in wordDict.items():
        wordDict.update({key: (1 + math.log10(value))})  # final tf values

    # id-f calculation
    for key in wordDict:  # key in dictionary
        if (getidf(key) == -1):
            multiply = math.log10(N)  # if getidf returns -1
        else:
            multiply = getidf(key)
        wordDict[key] = wordDict[key] * multiply  # from get-idf funtion
    # normalize
    sum = 0
    for key, value in wordDict.items():
        sum = sum + (value * value)
    sum = math.sqrt(sum)

    # normalize each value
    for key, value in wordDict.items():
        value = value / sum
        wordDict[key] = value  # update the values after normalization

    return (wordDict)


print(getqvec("The alternative, as cruz has proposed, is to deport 11 million people from this country"))


# cosine similarity
def query(queryString):
    wordDict = getqvec(queryString)

    # get tf-idf for each paragraph

    maxValue = 0
    cosineValue = 0
    counter = 0
    maxPara = ""  # getting the paragraph with the maximum cosine similarity
    paraDictionary = {}
    for i in list:
        sum = 0
        # dictionary for tf-idf of each paragraph
        paraDictionary = getqvec(i)
        counter += 1  # counter for tracking the paragraph number
        for key in wordDict:  # calculating cosine values by finding the same values in wordDict and paraDictionary
            if key in paraDictionary:
                cosineValue += wordDict[key] * paraDictionary[key]  # calculate the cosine value

        if (cosineValue > maxValue):  # find the max value
            maxValue = cosineValue
            para = counter  # for returning the right paragraph
        sum = 0
        cosineValue = 0

    if (maxValue == 0):
        return ("No match\n", 0.0000)  # if no match found in any paragraph
    else:
        return (paraDct[para] + "\n", maxValue)


print("%s%.4f" % query("The alternative, as cruz has proposed, is to deport 11 million people from this country"))
import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams
from itertools import permutations
import math
import string
import os
import csv
import pickle
import re
import random

#Global variables
word_sequences =[]
length = []
tail_end_is_space = []
head_start_is_space = []
tail_end_is_punct = []
head_start_is_punct = []
head = str()
tail = str()
head_words = []
tail_words = []
cluster=[]
text = []
cluster_list=[]
block_sequence_order = []
cluster_length=[]
last = 0
true_successor= []
testfile_name = "C:\\workspace\\mycorpus\\test-files\\test-9.txt"
csvfilename = "ADI_bigram_random_result_graph_test-9.csv"
probfilename = "ADI_bigram_random_probability-matrix-9.csv"
#Extracting Heads and Tails of all clusters and converting to words


infile = open(testfile_name,'rb')
file_stats = os.stat(testfile_name)
no_of_blocks = math.ceil( file_stats.st_size/4096 )
print("Number of blocks:",no_of_blocks)

block_sequence_order = random.sample(range(0,no_of_blocks),no_of_blocks)
#block_sequence_order = [3,10,11,6,1,8,7,0,9,13,12,4,14,2,5]


print("block sequence order :",block_sequence_order)
no_of_clusters = no_of_blocks
for i in range(0,no_of_clusters):
    cluster_list.append(bytes())

#Compute true_sucessors
for cno in range(0,no_of_clusters):
    if block_sequence_order[cno]+1 == no_of_clusters:
        true_successor.append(no_of_clusters)
        continue
    true_successor.append( block_sequence_order.index( block_sequence_order[cno]+1) )

#Reading clusters and store in a cluster list as per cluster sequence order
for i in range(0,no_of_clusters):
    infile.seek(i*4096)
    cluster_list[block_sequence_order.index(i)] = infile.read(4096)
    #cluster_length[block_sequence_order.index(i)]=len(cluster_list[block_sequence_order.index(i)]))
    
    
#Extracting Heads and Tails of all clusters and converting to words
for cno in range(0,no_of_clusters):
        
    #Extracting Head of a cluster and converting to words
    for i in range(0,60):
        if cluster_list[cno][i]<127:
            head += chr(cluster_list[cno][i])

    #check whether head start is space or punctuation
    if head[0] == ' ' :
        head_start_is_space.append(1)
    else:
        head_start_is_space.append(0)

    if (head[0] in string.punctuation) or (head[0].isspace()):
        head_start_is_punct.append(1)
    else:
        head_start_is_punct.append(0)
        
    head_words.append(nltk.wordpunct_tokenize(head))
    
    '''text =  nltk.wordpunct_tokenize(head)
    words = [w for w in text if w.lower() not in stopwords]
    head_words.append(words)'''
    
    #Extracting Tail of a cluster and converting to words
    l = len(cluster_list[cno])
    if l<4096:
        last = block_sequence_order[cno]
        print("last =", last)
    for j in range(l-60,l):
        if cluster_list[cno][j]<127:
            tail += chr(cluster_list[cno][j])
    
    #check whether tail end is space or punctuation
    if tail[len(tail)-1] == ' ' :
        tail_end_is_space.append(1)
    else:
        tail_end_is_space.append(0)

    if ( tail[len(tail)-1] in  string.punctuation ) or  ( tail[len(tail)-1].isspace() ) :
        tail_end_is_punct.append(1)
    else:
        tail_end_is_punct.append(0)

    tail_words.append(nltk.wordpunct_tokenize(tail))
    '''text =  nltk.wordpunct_tokenize(tail)
    words = [w for w in text if w.lower() not in stopwords]
    tail_words.append(words)'''
    length.append( len(tail_words[cno]) )

    
    #print(block_sequence_order[cno],":","HEAD:",head_words[cno],"\t\t\t TAIL:",tail_words[cno],end = '\n')
    head = tail = ""
infile.close()

#print("\nThe number of words in all tails(i.e length of each tail) is :", length, end = '\n')    

#Concatenate tail end word with head start word 
for i in range(0,no_of_clusters):
    #print("\ncluster:",i,"\n")
    #print("\nThe number of words in all tails(i.e length of each tail) is :", length, end = '\n')
    for j in range(0, no_of_clusters):       
        all_words = tail_words[i] + head_words[j]
        #print("Tail:",tail_words[i],"\t HEAD:",head_words[j],end = '\n')
        if tail_end_is_space[i] == 0 and head_start_is_space[j] == 0 and tail_end_is_punct[i] == 0 and head_start_is_punct[j] == 0:
            concat_word = all_words[length[i]-1] + all_words[length[i]]
            #print("Tail-Head concatenation word:",concat_word)
            del all_words[length[i]-1 : length[i]+1]
            all_words.insert(length[i]-1,concat_word)
        #print("Tail+Head:",all_words,end = '\n')
        word_sequences.append(all_words)
    


#Extracting the three words across the blocks of all combinations ( mid , mid-1, mid+1)
outfile = open("random-phrase.txt",'w')
j=0
outfile.write("cluster :"+str(block_sequence_order[j])+"\n")
for i in range(0,len(word_sequences)):
    if i%no_of_clusters == 0 and i!= 0:
        j = j+1
        outfile.write("cluster :"+str(block_sequence_order[j])+"\n")
    del word_sequences[i][:length[j]-4]
    del word_sequences[i][7:]
    outfile.write(str(word_sequences[i]))
    outfile.write('\n')


outfile.close()


#Using pre-trained language model and computing probability of word sequences


test_sentence_bigrams = [ list( ngrams(pad_both_ends(word_sequences[i], n=2),2) ) for i in range(len(word_sequences)) ] 
model_filepath = 'ADI_bigram_model.pkl'


# Load the pre-trained model using pickle
with open(model_filepath, 'rb') as file:
    lm_model = pickle.load(file)
    

prob_matrix = []
row = []

# Computing Probability Matrix
for i in range(len(test_sentence_bigrams)):
    product = 1
    for t in test_sentence_bigrams[i]:
        product = product*lm_model.score(t[1],[t[0]])
    row.append(product)
    
    if (i+1)% no_of_clusters == 0:
        prob_matrix.append(list(row))
        del row[:]


'''#printing the probability matrix        
print('\nThe Probability matrix is:\n')

for i in range(0,no_of_clusters):
    for j in range(0,no_of_clusters):
        print("%1.3e,"%(prob_matrix[i][j]), end = "  ")
    print("\n")'''

# writing the probability matrix to csv file  
with open(probfilename, 'w',newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)         
    # writing the data rows  
    csvwriter.writerows(prob_matrix)


rg = open(csvfilename, 'w',newline = '')
csvwriter = csv.writer(rg)
result=[]
'''paired = []
for i in range(0,no_of_clusters):
    paired.append(0)
#saving to file and printing the cluster number and index of max element in each row 
for i in range(0,no_of_clusters-1):
    row = prob_matrix[i]
    while(1):
        max_prob = max(row)
        max_index =  prob_matrix[i].index(max_prob)
        succ_cluster_no =  block_sequence_order[max_index]
        if paired[succ_cluster_no] == 0:
            paired[succ_cluster_no] = 1
            result.append([block_sequence_order[i],succ_cluster_no])
            csvwriter.writerow([block_sequence_order[i],succ_cluster_no])
            print("cluster",block_sequence_order[i],":",succ_cluster_no, end = "  ")
            print("\n")
            break
        else:
            row[max_index] = 0
            
rg.close()'''
csvwriter.writerow(["Cluster Number","Predicted Successor( cluster number)",
                    "True Successor( cluster number)","Block number", "Predicted successor(block number)"])
for i in range(0,no_of_clusters):
    row = prob_matrix[i]
    max_prob = max(row)
    max_index =  prob_matrix[i].index(max_prob)
    succ_cluster_no = max_index
    succ_block_no =  block_sequence_order[max_index]
    '''if block_sequence_order[i] == last:
        csvwriter.writerow([block_sequence_order[i],succ_cluster_no])
        continue'''
    result.append([block_sequence_order[i],succ_block_no])
    csvwriter.writerow([i,succ_cluster_no,true_successor[i],block_sequence_order[i],succ_block_no])
    print("Block:",block_sequence_order[i],"Successor:",succ_block_no, end = "  ")
    print("\n")

        


#calculating Accuracy
true_count = 0
for i in range(0,no_of_clusters):
    if result[i][1] == result[i][0]+1 or result[i][0] == last:
        true_count = true_count+1

accuracy = (true_count * 100)/(no_of_clusters)  
print("\npercentage of Accuracy = ", accuracy)
csvwriter.writerow(["Accuracy",accuracy])

rg.close()
    


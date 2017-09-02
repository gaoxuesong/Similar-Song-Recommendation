import sys
import pickle
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


#data,song_data,dictionary,embeddings = pickle.load(open(sys.argv[1], 'rb'))
data,song_data,embeddings = pickle.load(open(sys.argv[1], 'rb'))
sim_matrix = [[0]*len(song_data)]*len(song_data)


all_song_emd = []
songs = []
for song in song_data:
	emd = []
	for word_id in song_data[song]:
		emd.extend(embeddings[word_id])
	all_song_emd.append(emd)
	songs.append(song)
#print (all_song_emd)

for i in range(len(all_song_emd)):
	for j in range(len(all_song_emd)):
		if i != j and sim_matrix[i][j] == 0:
			dist = 1 - cosine_similarity(all_song_emd[i], all_song_emd[j]).reshape(1,-1)
			sim_matrix[i][j] = dist
			sim_matrix[j][i] = dist

#print (sim_matrix)
# print(data)
# print(songs)


#Testing
print ('**********Testing**********')
total = 0
for user in data:
	print ('User:')
	print (user)
	id_list = []
	test_song_id = songs.index(data[user][len(data[user])-1][0])
	# print(data[user][len(data[user])-1][0])
	# print(test_song_id)
	# print('lists')
	# print (data[user][0:len(data[user])-1])
	print ('Test Song')
	print (data[user][len(data[user])-1][0],test_song_id)
	print ('Train Songs List')
	for songs_list in data[user][0:len(data[user])-1]:
		print (songs_list[0],songs.index(songs_list[0]))
		id_list.append(songs.index(songs_list[0]))
	pred_song = [0]*len(all_song_emd)
	# print(id_list)
	# print(pred_song)
	
	for j in range(len(all_song_emd)):
		if j not in id_list:
			for each_id in id_list:
				# print(pred_song[each_id])
				# print(sim_matrix[each_id][j])
				pred_song[j] = pred_song[j] + sim_matrix[each_id][j]
				# print(pred_song[each_id])
	
	min_val = math.inf
	#print (pred_song)
	for each in pred_song :
		#print(each)
		if each != 0:
			#print("here")
			if each < min_val:
				min_val = each
	#print (min_val)
	closest_song = pred_song.index(min_val)
	#print(sim_matrix[closest_song])
	print ('Predicted Song for user')
	print(songs[closest_song],closest_song)
	print ('Accuracy of the Predicted Song with one removed song')
	acc = 1 - cosine_similarity(all_song_emd[test_song_id],all_song_emd[closest_song])
	acc = 100 - (acc*100)/2
	print (acc)
	print ('Accuracy of the Predicted Song with user playlist')
	avg_acc = 0
	for each in id_list:
		avg_acc = avg_acc + (1-cosine_similarity(all_song_emd[each],all_song_emd[closest_song]))
	avg_acc = avg_acc/len(id_list)
	avg_acc = 100 - (avg_acc*100)/2
	print(avg_acc)

	total = total + avg_acc


	print('-------------------')
print ('Model Accuracy')
print(total/len(data))





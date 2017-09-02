import codecs
import urllib.request, urllib.error, urllib.parse
import json
import socket
import pylast
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.datasets import fetch_20newsgroups
import codecs
from sklearn.feature_extraction.text import CountVectorizer

apikey_musixmatch = '949dd5ef51c7a0903785642dd4202303'
apiurl_musixmatch = 'http://api.musixmatch.com/ws/1.1/'

API_KEY = "42f59330010037402ee13cb50711438b"  # this is a sample key
API_SECRET = "fe57fd1fa32dbbee290bb3d65138eed4"

# In order to perform a write operation you need to authenticate yourself
username = "gthrishivakumar"
password_hash = pylast.md5("F@mily33")

network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET,
                               username=username, password_hash=password_hash)

unique_tracks = {}
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

vect = CountVectorizer(tokenizer=tokenize, stop_words='english')

def parse_lyrics(data):
    songdata = []
    # print("parsing \n", len(data))
    line = (data.split("\n"))

    i = 0
    # print("****************************",i)
    while not (line[i] == "..."):
        songdata.append(line[i])
        songdata.append(" ")
        i += 1
    songdata = "".join(songdata);
    return songdata


def song_lyric(song_name, artist_name):
    while True:
        querystring = apiurl_musixmatch + "matcher.lyrics.get?q_track=" + urllib.parse.quote(
            song_name) + "&q_artist=" + urllib.parse.quote(
            artist_name) + "&apikey=" + apikey_musixmatch + "&format=json&f_has_lyrics=1"
        # matcher.lyrics.get?q_track=sexy%20and%20i%20know%20it&q_artist=lmfao
        request = urllib.request.Request(querystring)
        # request.add_header("Authorization", "Bearer " + client_access_token)
        request.add_header("User-Agent",
                           "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)")  # Must include user agent of some sort, otherwise 403 returned
        while True:
            try:
                response = urllib.request.urlopen(request,
                                                  timeout=4)  # timeout set to 4 seconds; automatically retries if times out
                raw = response.read()
            except socket.timeout:
                print("Timeout raised and caught @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
                continue
            break
        json_obj = json.loads(raw.decode('utf-8'))
        # print("&&&&& obj &&&&", json_obj["message"]["body"])
        if (json_obj["message"]["body"] != [] and json_obj["message"]["body"] != ""):
            # print("empty")
            body = json_obj["message"]["body"]["lyrics"]["lyrics_body"]
            copyright = json_obj["message"]["body"]["lyrics"]["lyrics_copyright"]
            if (body != "" and copyright != ""):
                return (body + "\n\n" + copyright)
            else:
                return "NotFound"
        else:
            return "NotFound";
            # lyrics_tracking(tracking_url)


def lyrics_tracking(tracking_url):
    while True:

        try:
            querystring = tracking_url
            request = urllib.request.Request(querystring)
            request.add_header("User-Agent",
                               "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)")  # Must include user agent of some sort, otherwise 403 returned

            response = urllib.request.urlopen(request,
                                              timeout=4)  # timeout set to 4 seconds; automatically retries if times out
            raw = response.read()
        except socket.timeout:
            print("Timeout raised and caught #############################################")
            continue
        break
        # print(raw)


def get_lyrics(song_id):
    artist = unique_tracks[song_id][0];
    title = unique_tracks[song_id][1].split("\n")[0];

    #print("***************************************************************************** \n ", song_id, "\n", artist,"\n", title, "\n", song_lyric(title,artist))
    song_lyric_value = song_lyric(title, artist)
    #song_lyric_value= song_lyric("You're The One","Dwight Yoakam")
    #print("*** ", song_lyric_value)
    if(song_lyric_value != "NotFound" and song_lyric_value != ""):
        lyrics= parse_lyrics(song_lyric_value)
    else:
        lyrics = "NotFound"
    return lyrics
    # return "yes"


def parse_unique_tracks():
    unique_track_file = open("unique_tracks.txt", 'r', encoding='utf-8');
    for line in unique_track_file:
        line = line.split('<SEP>');
        unique_tracks[line[1]] = (line[2], line[3]);


def get_tags(song_id):
    artist = unique_tracks[song_id][0];
    title = unique_tracks[song_id][1].split('\n')[0];

    try:
        track = network.get_track(artist, title)
        # print("\n $$$$$$$$$$$$$$$ \n", artist, ", ", title, "\n")
        tags_list = track.get_top_tags();
        if (tags_list != []):
            list_tags = parse_tags_list(tags_list)
            return list_tags
        else:
            return "NO TAGS"
    except pylast.WSError as err:
        print("TRACK not found")
        return "NO TAGS"


def get_data():
    parse_unique_tracks();
    infile = "kaggle_visible_evaluation_triplets.txt"
    outfile = "user_song_data_final_2.txt";

    source = open(infile, 'r', encoding= 'utf-8')
    target = open(outfile, 'w' , encoding='utf-8')

    for line in source:
        line = line.split("\t");
        header = unique_tracks[line[1]];
        artist = header[0];
        title = header[1].split("\n")[0];

        target.write(line[0] + ',');
        target.write(line[1] + ',')
        target.write(artist+',')
        target.write(title+',')
        lyrics = get_lyrics(line[1])
        if (lyrics != "NotFound" and lyrics != ""):
            try:
                vect.fit([lyrics]);
                lyrics = vect.get_feature_names()
                for token in lyrics:
                    target.write(token + ',')
            except ValueError:
                pass


        tags = get_tags(line[1])
        if (tags != "NO TAGS" and tags != []):
            tagsStr= ''.join(tags)
            #print("tags string", tagsStr)
            try:
                vect.fit([tagsStr])
                tagsStr= vect.get_feature_names()
                #print("tag featires", tagsStr)
                for tok in tagsStr:
                    #print("token", tok)
                    target.write(tok + ',')
            except ValueError:
                pass

        target.write('\n');

    source.close()
    target.close()


def parse_tags(data):
    fullTag = ""
    for i in range(0, len(data)):
        fullTag += str(data[i][0])
        fullTag += "\t"
        fullTag+= "\n"
    return fullTag


def parse_tags_list(data):
    list_tags = [len(data)]
    fulltag = parse_tags(data)
    list_tags = fulltag.split("\n")
    return list_tags


if __name__ == "__main__":
    get_data();




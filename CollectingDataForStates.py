states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]

candidates = ["Bernie Sanders", "Elizabeth Warren", "Kamala Harris", "Joe Biden", "Pete Buttigieg"]
import subprocess
import os
save_path="/home/kapil/PycharmProjects/Political-Opinion-Mining/Candidate_Tweets_per_State/"
# twitterscraper "Bernie Sanders near:California" -bd 2019-09-23 -ed 2019-11-23 -o Bernie_California.json
# os.system("twitterscraper trump -bd 2019-09-23 -ed 2019-11-23 -l 10 -o "+save_path+"Bernie_California.json")
# subprocess.run(["ls", "-l"])
for candidate in candidates:
  for word in states:
    query = 'twitterscraper "'+candidate+' near:' + word + '" -bd 2019-09-23 -ed 2019-11-23 -o '+save_path+candidate.replace(" ","")+'_' +word+ '.json'
    print(query)
    os.system(query)
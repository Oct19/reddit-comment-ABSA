import praw
import pandas as pd
import nltk # nltk for NLP
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Scraping
reddit = praw.Reddit(client_id="r64_mFOCNn5LtVQvZJwBPw",		 # your client id
							client_secret="s80A58yLhPe6AkIfNh05q08t1CfROQ",	 # your client secret
							user_agent="Scraping v1.0 by DingDing")	 # your user agent
subreddit_name = 'Relationship_advice'
headlines = set() # Use set to prevent adding duplicates

for submission in reddit.subreddit(subreddit_name).hot(limit=100):
    # submission contains ID, author, title, score, upvote_ration, etc
    headlines.add(submission.title)
print(len(headlines))

# NLP
sia = SIA()
results = []
for line in headlines:
    # pol_score contain sentiment values and texts, it looks like this:
    # 'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0, 'headline': 'My (21M) GF (20F) wants to live apart.'
    pol_score = sia.polarity_scores(line) # -> dict
    pol_score['headline'] = line
    results.append(pol_score)

df = pd.DataFrame.from_records(results)
df['label'] = 0 # assign label based on sentiment values
df.loc[df['compound'] > 0.3, 'label'] = 1 # positive sentiment
df.loc[df['compound'] < -0.3, 'label'] = -1 # negative sentiment

# Save results
outdir = './result/Headlines sentiment'
if not os.path.exists(outdir):
    os.mkdir(outdir)

str_current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
outname = subreddit_name+' '+str_current_datetime
df2 = df[['headline', 'label']]
fill_name = os.path.join(outdir, outname) 
df2.to_csv(fill_name+'.csv', encoding = 'utf-8', index=False)

# print('Positive headlines:\n')
# print(list(df[df['label'] == 1].headline)[:5])
# print('Negative headlines:\n')
# print(list(df[df['label'] == -1].headline)[:5])

fig, ax = plt.subplots(figsize=(10,10))
counts = df.label.value_counts(normalize=True) * 100 # percentage
sns.barplot(x=counts.index, y=counts, ax = ax)
ax.set_xticklabels(['Neg', 'Neutral', 'Pos'])
ax.set_ylabel('Percentage')
plt.savefig(fill_name+'.png')
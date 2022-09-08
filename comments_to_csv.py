import praw
import pandas as pd
import nltk # nltk for NLP
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

subreddit_name = 'Solidworks'
outdir = './result/Comments ABSA' # -> output directory
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Scraping
reddit = praw.Reddit(client_id="r64_mFOCNn5LtVQvZJwBPw",		 # your client id
							client_secret="s80A58yLhPe6AkIfNh05q08t1CfROQ",	 # your client secret
							user_agent="Scraping v1.0 by DingDing")	 # your user agent

subreddit = reddit.subreddit(subreddit_name)
data = []

for submission in subreddit.top(limit=5):
    print('Title: {}, ups: {}'.format(submission.title, submission.ups))

    submission.comments.replace_more(limit=0)
    print(len(submission.comments.list()))

    for comment in submission.comments.list():
        comment_dict = {
            'Submission ID': comment.link_id,
            'Comment ID': comment.id,
            'Time UTC': comment.created_utc, 
            'Upvotes': comment.score,
            'Permalink': comment.permalink,
            'Body': comment.body
        }
        if not comment.stickied:
            data.append(comment_dict)
print(len(data))
df = pd.DataFrame(data)

str_current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
outname = subreddit_name+str_current_datetime
fill_name = os.path.join(outdir, outname) 
df.to_csv(fill_name+'.csv', encoding = 'utf-8', index=False)

import tweepy
import pandas as pd
import argparse

cons_key = 'xxx'
cons_secret = 'xxx'
token = 'xxx'
token_secret = 'xxx'

auth = tweepy.OAuthHandler(cons_key, cons_secret)
auth.set_access_token(token, token_secret)
tweeter = tweepy.API(auth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, help='location of input file txt file')
    args = parser.parse_args()

    if args.infile:
        filename = args.infile
    else:
        filename = 'BIDU.txt'

    df = pd.read_csv(filename, sep='\t', header=0, usecols=[0, 1])

    tweet_text = []
    for index, row in df.iterrows():
        try:
            mytext = tweeter.get_status(row['TweetID'])
            mytext= mytext.text
        except:
            mytext = 'TweetID not found'

        tweet_text.append(mytext)
        print(mytext)

    df['TweetText'] = tweet_text

    print(df.head())

# check if result id is ok vs request
# add restart from... if crash

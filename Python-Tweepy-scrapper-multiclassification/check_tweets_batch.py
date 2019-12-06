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
    parser.add_argument('--infile', type=str, default='BIDU.txt', help='location of input file txt file')
    parser.add_argument('--outfile', type=str, default='output.csv', help='location of output file txt file')
    parser.add_argument('--batch_size', type=int, default=100, help='size of batch to proceed max=100')
    args = parser.parse_args()

    if args.batch_size > 100:
        batchsize = 100
    else:
        batchsize = args.batch_size

    filename = args.infile
    outfilename = args.outfile

    df = pd.read_csv(filename, sep='\t', header=0, usecols=[0, 1])

    tweet_batch = []
    for index, row in df.iterrows():
        tweet_batch.append(row['TweetID'])
        if len(tweet_batch) == batchsize or index + 1 == df.shape[0]:
            print(index + 1)
            # read tweets
            result = tweeter.statuses_lookup(tweet_batch, map_=True)
            result_text = []
            for r in result:
                try:
                    df.loc[df['TweetID'] == r.id, 'tweet'] = r.text
                except:
                    df.loc[df['TweetID'] == r.id, 'tweet'] = 'ERROR'
            tweet_batch = []

    print(df.head())
    df.to_csv(outfilename, sep=';', index=False)

# add restart from... if crash

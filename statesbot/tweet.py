import re
import pickle

import tweepy

from .SECRET import (
    my_consumer_key,
    my_consumer_secret,
    my_access_token,
    my_access_token_secret,
)

USER_ID = "bot_states"


def get_api():
    my_auth = tweepy.OAuthHandler(my_consumer_key, my_consumer_secret)
    my_auth.set_access_token(my_access_token, my_access_token_secret)
    return tweepy.API(my_auth)


def current_tweet_id():
    tweets = get_api().user_timeline(
        screen_name=USER_ID, count=10, include_rts=False, tweet_mode="extended"
    )
    for t in tweets:
        mat = re.match(r"Map (\d+):.*", t.full_text)
        if mat:
            return 1 + int(mat.group(1))
    return 1


def tweet_map(title, image):
    get_api().update_with_media(image, status=title)
    print("Tweeted message")

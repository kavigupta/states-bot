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
    my_auth = tweepy.OAuthHandler(
        my_consumer_key,
        my_consumer_secret,
    )
    my_auth.set_access_token(my_access_token, my_access_token_secret)
    return tweepy.API(
        my_auth,
        retry_count=10,
        retry_delay=5,
        retry_errors=set([503]),
    )


def current_tweet_id():
    tweets = get_api().user_timeline(
        screen_name=USER_ID, count=10, include_rts=False, tweet_mode="extended"
    )
    for t in tweets:
        mat = re.match(r"Map (\d+):.*", t.full_text)
        if mat:
            return 1 + int(mat.group(1))
    return 1


def tweet_map(title, extra_title, images):
    [atlas_key] = {key for key, _ in images if key.startswith("atlas")}

    initial_tweet = get_api().update_with_media(
        images[atlas_key, "small"],
        status=title + " " + extra_title % images[atlas_key, "large"],
    )
    print("Tweeted message")
    if ("politics", "small") not in images:
        return
    get_api().update_with_media(
        images["politics", "small"],
        in_reply_to_status_id=initial_tweet.id,
        status="Political Version " + extra_title % images["politics", "large"],
    )
    print("Tweeted reply")

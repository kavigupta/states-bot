import os
import subprocess
import tqdm
import pickle
import functools

from statesbot.data import Data
from statesbot.version import version
from statesbot.run_sampler import sample_guaranteed
from statesbot.tweet import tweet_map, current_tweet_id


def get_n_states(seed):
    if seed % 3 == 1:
        return 50
    return np.random.RandomState(seed).randint(40) + 10


def guarantee_path(seed):
    path = f"maps/{seed}"
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path

@functools.lru_cache(None)
def get_data():
    return Data()

def generate_map(seed):
    path = guarantee_path(seed)
    path = f"{path}/assignment.pkl"
    if os.path.exists(path):
        return path
    n_states = get_n_states(seed)
    data = get_data()
    assign = sample_guaranteed(data, rng_seed=seed, n_states=n_states, pbar=tqdm.tqdm)
    title = f"Map {seed}: {n_states} states by Equipopulation"
    out = dict(map=assign.export(data), version=version, title=title)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return path


def render_map(seed):
    data = get_data()
    with open(generate_map(seed), "rb") as f:
        map_object = pickle.load(f)
        map_object["map"].attach_to(data)

    map_object["map"].ship()
    path = {}
    for which in "atlas", "politics":
        for size, dpi in ("small", 130), ("large", 340):
            print(which, size, dpi)
            path[which, size] = f"maps/{seed}/states_bot_{seed:02d}_{size}_{which}.png"
            if os.path.exists(path[which, size]):
                continue
            subprocess.check_call(
                [
                    "/usr/bin/python3",
                    "export.py",
                    map_object["title"],
                    path[which, size],
                    map_object["version"],
                    str(dpi),
                    which,
                    *(str(x) for x in map_object["map"].statistics())
                ]
            )
    return map_object, path
def run_bot(seed):

    map_object, path = render_map(seed)

    subprocess.check_call(["git", "add", "maps"])
    subprocess.call(["git", "commit", "-m", f"add images for seed={seed}"])
    subprocess.check_call(["git", "push", "origin", "master"])
    tweet_map(
        map_object["title"],
        f"(full size: https://github.com/kavigupta/states-bot/raw/master/%s)",
        path,
    )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render-map-only", action="store_true")

    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = current_tweet_id()
    if args.render_map_only:
        render_map(seed)
    else:
        run_bot(seed)

if __name__ == "__main__":
    main()

import os
import subprocess
import tqdm
import pickle
import functools

import numpy as np
from statesbot.geographies.north_america import NorthAmericaDataset

from statesbot.version import version
from statesbot.run_sampler import sample_guaranteed
from statesbot.tweet import tweet_map, current_tweet_id
from statesbot.geographies.usa import USACountiesDataset
from statesbot.geographies.euroamerica import EuroAmericaCountiesDataset
from statesbot.geographies.europe import EuropeCountiesDataset
from statesbot.geographies.canada import CanadaDataset


def get_n_states(seed, max_states):
    if seed % 3 == 1:
        return max_states
    return np.random.RandomState(seed).choice(np.arange(2, max_states))


def guarantee_path(seed):
    path = f"maps/{seed}"
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path


def get_geography(seed):
    schedule = [
        EuroAmericaCountiesDataset,
        NorthAmericaDataset,
        USACountiesDataset,
        EuropeCountiesDataset,
        CanadaDataset,
        USACountiesDataset,
    ]
    return schedule[seed % len(schedule)]().construct()


map_types = {
    "atlas": 165,
    "politics": 165,
    "atlas_europe": 165,
    "atlas_north_america": 165,
    "atlas_euroamerica": 130,
    "atlas_canada": 165,
}


def generate_map(seed):
    path = guarantee_path(seed)
    path = f"{path}/assignment.pkl"
    if os.path.exists(path):
        return path
    geography = get_geography(seed)
    n_states = get_n_states(seed, geography.max_states)
    assign = sample_guaranteed(geography, rng_seed=seed, n_states=n_states)
    title = f"Map {seed}: {n_states} states by Equipopulation"
    out = dict(map=assign.export(), version=version, title=title)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return path


def render_map(seed):
    geography = get_geography(seed)
    with open(generate_map(seed), "rb") as f:
        map_object = pickle.load(f)
        map_object["map"].attach_to(geography)

    map_object["map"].ship()
    path = {}
    for which in geography.atlas_types:
        for size, dpi in ("small", map_types[which]), ("large", 320):
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
                    *(str(x) for x in map_object["map"].statistics()),
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

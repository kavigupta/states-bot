import numpy as np


def combine_names(a, b):
    vowels = set("aeiouy")
    common_letters = set(a[len(a) // 2 :]) & set(b[: len(b) // 2])
    if not common_letters:
        if set(b) & vowels:
            while not b[0].lower() in vowels:
                b = b[1:]
            b = b[1:]
        if set(a) & vowels:
            while not a[-1].lower() in vowels:
                a = a[:-1]
        return a + b
    best_indices = float("inf"), float("inf")
    for c in common_letters:
        idx = a[::-1].index(c), b.index(c)
        if sum(idx) < sum(best_indices):
            best_indices = idx
    return a[: -best_indices[0] - 1] + b[best_indices[1] :]


def name_by_regions(regions, population, counties):
    contained_states = []
    for state, scounties in regions.items():
        overlap = list(set(counties) & set(scounties))
        overlap = population[overlap].sum()
        if overlap / population[scounties].sum() > 2 / 3:
            contained_states.append((overlap, state))
    contained_states = sorted(contained_states, reverse=True)
    if sum(c[0] for c in contained_states[:2]) < population[list(counties)].sum() / 2:
        return None
    if len(contained_states) == 1:
        return contained_states[0][1]
    (_, s1), (_, s2), *_ = contained_states
    return combine_names(s1, s2)


def name_state(regions, population, cities, counties):
    name = name_by_regions(regions, population, counties)
    if name is not None:
        return name

    cities_for_state = [
        city for county in counties for _, city in cities[county].iterrows()
    ]
    biggest_city = max(cities_for_state, key=lambda x: x.Population)
    if len(counties) == 1:
        return biggest_city["ASCII Name"]
    cities_for_state = sorted(
        [x for x in cities_for_state if not (x == biggest_city).all()],
        key=lambda x: x.Population,
    )[-5:]
    distances = [
        (x.x - biggest_city.x) ** 2 + (x.y - biggest_city.y) ** 2
        for x in cities_for_state
    ]

    distances = np.array(distances) / max(distances)
    next_city = max(
        zip(cities_for_state, distances),
        key=lambda x: x[1],
    )[0]
    return combine_names(biggest_city["ASCII Name"], next_city["ASCII Name"])

import simplejson as json

import ijson
def flatten(x):
    if isinstance(x, list):
        if isinstance(x[0][0], list):
            return flatten([z for y in x for z in y])
        return np.array([[float(x), float(y)] for x, y, *_ in x])

    return flatten(x["coordinates"])

f = open("/home/kavi/Downloads/precincts-with-results.geojson")

pbar = tqdm.tqdm(total=os.stat(f.name).st_size)
summarized_geojsons = []
for x in ijson.items(f, "features.item"):
    summarized_geojsons.append(
        {**x["properties"], "centroid": flatten(x["geometry"]).mean(0).tolist()}
    )
    pbar.update(f.tell() - pbar.n)
    pbar.refresh()
f.close()

with open("precinctdata/precinctdata.json", "w") as f:
    json.dump(summarized_geojsons, f, use_decimal=True)

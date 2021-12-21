import numpy as np
import knapsack

from statesbot.data import Data, Metadata
from statesbot.assign import Assignment


data = Data()
counties = {c.ident : (c.dem_2020 * c.pop, c.pop) for c in data.countylikes if c.state == "California"}
to_consider = list(counties)
redca = []
for f, c in list(counties.items()):
    if c[0] < 0:
        redca.append((f, c))
        to_consider.remove(f)
total_margin = sum(x[1][0] for x in redca)
redca = [x[0] for x in redca]
size = [counties[k][0] for k in to_consider]
weight = [counties[k][1] for k in to_consider]
_, idxs = knapsack.knapsack(size, weight).solve(-total_margin)
redca += [to_consider[i] for i in idxs]
blueca = [x for x in counties if x not in redca]
mapback = {c.ident : i for i, c in enumerate(data.countylikes)}
county_to_state = np.zeros(len(data.countylikes), dtype=np.int)
county_to_state[[mapback[x] for x in blueca]] = 1
county_to_state[[mapback[x] for x in redca]] = 2
assign = Assignment.from_county_to_state(data, Metadata(3, data.pops, data, data.centers), county_to_state)
map = assign.export(data)
map.attach_to(data)
map.ship()

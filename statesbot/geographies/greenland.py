import geopandas

from .geography import GeographySource


class GreenlandDataset(GeographySource):
    def version(self):
        return "1.0.7"

    def geo_dataframe(self):
        df = geopandas.read_file("shapefiles/greenland/GRL_adm0.shp")
        df.geometry = df.geometry.simplify(tolerance=0.002)
        df = df[["geometry"]]
        df["ident"] = "GREENLAND"
        df["population"] = 56367
        df["dem_2020"] = 0
        df["name"] = "Greenland"
        return df

    def additional_edges(self):
        return []

    def regions(self, geodb):
        return {"Greenland": [0]}

    def atlas_types(self):
        return []

    def max_states(self):
        return 0

from collections import defaultdict

import geopandas
import pandas as pd
import pycountry

from ..utils import sequence, to_all

from .geography import GeographySource

MANUAL_POPULATIONS = {
    "Svalbard": 2939,
    "Jan Mayen": 0,
    "Arr. La Louvière": 80637,
    "Bjelovarsko-bilogorska županija": 119764,
    "Virovitičko-podravska županija": 84836,
    "Požeško-slavonska županija": 78034,
    "Brodsko-posavska županija": 158575,
    "Osječko-baranjska županija": 305032,
    "Vukovarsko-srijemska županija": 180117,
    "Karlovačka županija": 128899,
    "Sisačko-moslavačka županija": 172439,
    "Grad Zagreb": 790017,
    "Međimurska županija": 113804,
    "Varaždinska županija": 175951,
    "Koprivničko-križevačka županija": 115584,
    "Krapinsko-zagorska županija": 132892,
    "Zagrebačka županija": 317606,
    "City of Belgrade": 1694480,
    "Zapadnobačka oblast": 168841,
    "Južnobanatska oblast": 275289,
    "Južnobačka oblast": 618624,
    "Severnobanatska oblast": 133934,
    "Severnobačka oblast": 177044,
    "Srednjobanatska oblast": 171988,
    "Sremska oblast": 295132,
    "Zlatiborska oblast": 262644,
    "Kolubarska oblast": 160588,
    "Mačvanska oblast": 274549,
    "Moravička oblast": 196516,
    "Pomoravska oblast": 194676,
    "Rasinska oblast": 219017,
    "Raška oblast": 303552,
    "Šumadijska oblast": 278917,
    "Borska oblast": 109210,
    "Braničevska oblast": 163058,
    "Zaječarska oblast": 104352,
    "Jablanička oblast": 196265,
    "Nišavska oblast": 357920,
    "Pirotska oblast": 82537,
    "Podunavska oblast": 182895,
    "Pčinjska oblast": 195041,
    "Toplička oblast": 82067,
    "Belfast": 341877,
    "Antrim and Newtownabbey": 142492,
    "Ards and North Down": 160864,
    "Armagh City, Banbridge and Craigavon": 214090,
    "Causeway Coast and Glens": 144246,
    "Derry City and Strabane": 150679,
    "Fermanagh and Omagh": 116835,
    "Lisburn and Castlereagh": 144381,
    "Mid and East Antrim": 138773,
    "Mid Ulster": 147392,
    "Newry, Mourne and Down": 180012,
}

REMAPPED = {
    "BE224": "BE221",
    "BE225": "BE222",
    "BE328": "BE327",
    "BE32A": "BE321",
    "BE32B": "BE322",
    "BE32C": "BE325",
    "BE32D": "BE326",
    "EE009": "EE006",
    "EE00A": "EE007",
    "ITG2D": "ITG25",
    "ITG2E": "ITG26",
    "ITG2F": "ITG27",
    "ITG2G": "ITG28",
    "ITG2H": "ITG29",
    "NO081": "NO011",
    "NO0A1": "NO051",
    "NO0A2": "NO052",
    "NO0A3": "NO053",
    "UKK24": "UKK21",
    "UKK25": "UKK22",
}

TO_REMOVE = ["FRY10", "FRY20", "FRY30", "FRY40", "FRY50"]


def europe_nuts3_dataset():
    df = geopandas.read_file(
        "shapefiles/europe_counties/NUTS_RG_03M_2021_3035_LEVL_3.shp"
    ).to_crs(epsg=4326)
    stats = pd.read_csv("csvs/euro_census_output.csv", sep=",")
    stats = stats[stats.TIME == 2016]
    stats_map = {x: int(y) if y != ":" else ":" for x, y in zip(stats.GEO, stats.Value)}
    stats_map = {x: y for x, y in stats_map.items() if not x.startswith("UKN")}
    subtracts = {"BE323": 80637}
    assert set(stats) & set(REMAPPED.values()) == set()
    MANUAL_POPULATIONS["Innlandet"] = stats_map.pop("NO021") + stats_map.pop("NO022")
    MANUAL_POPULATIONS["Trøndelag"] = stats_map.pop("NO061") + stats_map.pop("NO062")
    MANUAL_POPULATIONS["Troms og Finnmark"] = stats_map.pop("NO072") + stats_map.pop(
        "NO073"
    )
    MANUAL_POPULATIONS["Viken"] = (
        stats_map.pop("NO012") + stats_map.pop("NO032") + stats_map.pop("NO031")
    )
    MANUAL_POPULATIONS["Vestfold og Telemark"] = stats_map.pop("NO033") + stats_map.pop(
        "NO034"
    )
    MANUAL_POPULATIONS["Agder"] = stats_map.pop("NO041") + stats_map.pop("NO042")
    remaining = list(stats_map)
    missing = []
    result = []
    for _, row in df.iterrows():
        x = row.NUTS_ID
        x = REMAPPED.get(x, x)
        if x in stats_map and stats_map[x] != ":":
            result.append(int(stats_map[x]) - subtracts.get(x, 0))
            remaining.remove(x)
            assert row.NAME_LATN not in MANUAL_POPULATIONS
            continue
        if row.NAME_LATN in MANUAL_POPULATIONS:
            result.append(MANUAL_POPULATIONS[row.NAME_LATN])
            continue
        missing.append((row.NUTS_ID, row.NAME_LATN))
    df["population"] = result
    df["dem_2020"] = 0
    df = df.rename(columns={"NUTS_ID": "ident"})

    df = df[df.ident.apply(lambda x: x not in TO_REMOVE)]
    return df


class EuropeCountiesDataset(GeographySource):
    def version(self):
        return "1.0.12"

    def geo_dataframe(self):
        return europe_nuts3_dataset()

    def additional_edges(self):
        greece = [
            # Pelopenesian Peninsula -> Crete -> Kalymnos
            ("EL307", "EL434"),
            ("EL432", "EL421"),
            # Lefkada and Ithaki to mainland
            ("EL624", "EL631"),
            ("EL623", "EL631"),
            ("EL624", "EL623"),
            # Cyprus to Greek Isalnds, Turkey
            ("CY000", "EL421"),
            ("CY000", "TR611"),
            ("CY000", "TR622"),
            # Evros -> Lesvos -> Chivos -> Ikaria -> Kalymnos
            *sequence("EL511", "EL411", "EL413", "EL412", "EL421"),
            # Lesvos -> Turkey
            ("EL411", "TR222"),
            ("EL411", "TR221"),
            # Chios -> Turkey
            ("EL413", "TR310"),
            # Ikaria -> Turkey
            ("EL412", "TR321"),
            # Kalymnos -> Turkey
            ("EL421", "TR323"),
            # Andros -> Ikaria, Kalymos
            ("EL422", "EL412"),
            ("EL422", "EL421"),
            # Andros -> Greece
            ("EL422", "EL305"),
            ("EL422", "EL642"),
            # Zakynthos -> Greece
            ("EL621", "EL633"),
            # Zakynthos -> Ithaki
            ("EL621", "EL623"),
            # Kerkyra --> Mainland
            ("EL622", "EL542"),
            ("EL622", "AL035"),
        ]
        denmark = [
            # copenhagen to mainland
            ("DK031", "DK022"),
            # sweden to copenhagen
            ("SE224", "DK013"),
            ("SE224", "DK012"),
            ("SE224", "DK011"),
            # Bornholm to Sweden
            ("DK014", "SE224"),
        ]
        scotland_ireland = [
            # Scotland to Northern Ireland
            ("UKM63", "UKN0C"),
            ("UKM94", "UKN0F"),
            ("UKM92", "UKN0F"),
            # Scotland islands to Iceland
            ("UKM64", "IS002"),
            # Scotland islands to Mainland
            ("UKM63", "UKM64"),
        ]
        italy = [
            # Corsica to Italy
            ("FRM02", "ITI16"),
            # Corsica to Sardinia
            ("FRM01", "ITG2D"),
            # Italy to Sardinia
            ("ITF65", "ITG13"),
            # Malta to Italy
            *to_all("ITG18", "MT001", "MT002"),
        ]
        spain = [
            # Eivissa --> Mainland
            ("ES531", "ES521"),
            ("ES531", "ES523"),
            # Eivissa --> Mallorca --> Menorca
            *sequence("ES531", "ES532", "ES533"),
            # Ceuta --> Cadiz
            ("ES630", "ES612"),
            # Melilla --> Almeria, Granada
            ("ES640", "ES611"),
            ("ES640", "ES614"),
            # Canary Islands east to west
            *sequence("ES703", "ES707", "ES706", "ES709", "ES705", "ES704", "ES708"),
            ("ES703", "ES706"),
            # Canary islands -> Maderia -> Algarve
            *sequence("ES707", "PT300", "PT150"),
            # Acores --> Maderia
            ("PT200", "PT300"),
        ]
        return [
            ## GREECE
            *greece,
            *denmark,
            *scotland_ireland,
            *italy,
            *spain,
            # Finland to Estonia
            ("FI1B1", "EE001"),
            # Aaland to Finland mainland
            ("FI200", "FI1C1"),
            # Chunnel
            ("UKJ44", "FRE12"),
            # Jan Mayen --> Iceland
            ("NO0B1", "IS002"),
            # Svaldbart --> Norway
            ("NO0B2", "NO074"),
            # Gotland -> Mainland
            *to_all("SE214", "SE123", "SE213"),
            # Isle of Wight
            ("UKJ34", "UKJ36"),
            # Isle of Anglesey
            ("UKL11", "UKL12"),
            # Scotland -> Orkney -> Shetland -> Norway
            *sequence("UKM61", "UKM65", "UKM66", "NO0A2"),
        ]

    def regions(self, geodb):
        countries = {}
        for country in pycountry.countries:
            countries[country.alpha_2] = country.name
        countries["EL"] = countries.pop("GR")
        countries["UK"] = countries.pop("GB")
        by_state = defaultdict(list)
        for i, (_, row) in enumerate(geodb.iterrows()):
            by_state[countries[row.CNTR_CODE]].append(i)
        return dict(by_state.items())

    def atlas_types(self):
        return ["atlas_europe"]

    def max_states(self):
        return 25

"""
Loaders

Raw table loaders.
Loaders load by raw tables, with minimal/sensible processing.
"""
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pooch


from ._version import __version__




registry_path = files("tableaux.data").joinpath("registry.txt")

PUP = pooch.create(
    path=pooch.os_cache("tableaux"),
    base_url="https://github.com/remrama/tableaux/raw/{version}/src/tableau/data/",
    version=__version__,
    version_dev="main",
    registry=None,
)

PUP.load_registry(registry_path)


#########################################
def _list_available_tables(dataset):
    """List available tables for a given dataset.

    dataset: str
        Name of dataset
    """
    assert isinstance(dataset, str), "`dataset` must be a string"
    glob = sorted(p.stem for p in data_dir.joinpath(dataset).glob("*.tsv"))
    return sorted(glob)

def _compile_table_filepath(dataset, table):
    """
    Compile a filepath for a given dataset and table combination.
    dataset: str
        Name of dataset
    table: str
        Name of table
    """
    assert isinstance(dataset, str), "`dataset` must be a string"
    assert isinstance(table, str), "`table` must be a string"
    available_tables = _list_available_tables(dataset)
    assert table in available_tables, f"{table} must be one of {available_tables}"
    return data_dir.joinpath(dataset).joinpath(table).with_suffix(".tsv")



def load_barrett2020(table):
    """
    Load table from Barrett 2020, Dreaming.
    Dreams About COVID-19 Versus Normative Dreams: Trends by Gender
    doi: 10.1037/drm0000149

    table: str
        Name of desired table.
        Available tables are Table1 (table1) and Table2 (table2)

    Available tables.
    Table 1: Female Pandemic Survey Dreams Versus Hall and van de Castle Female Normative Dreams
    Table 2: Male Pandemic Survey Dreams Versus Hall and van de Castle Male Normative Dream[s]
    Manual: Manually extracted features.
    """
    fp = _compile_table_filepath("barrett2020", table)
    df = pd.read_table(fp)
    if table != "manual":
        # Modify values
        df["p"] = df["p"].str.rstrip("*")
        df["LIWC category and content examples"] = (
            df["LIWC category and content examples"]
            .str.split(":").str[0]
            .str.lower()
            .replace(
                {
                    "biological processes": "bio",
                    "negative emotions": "negemo",
                    "positive emotions": "posemo",
                    "sadness": "sad",
                }
            )
        )
        # Set data types
        df = df.astype(
            {
                "Pandemic M": float,
                "Pandemic SD": float,
                "Normative M": float,
                "Normative SD": float,
                "t": float,
                "p": str,
                "LIWC category and content examples": str,
            })
        df["LIWC category and content examples"] = pd.Categorical(df["LIWC category and content examples"], ordered=False)
        # Set, sort, and rename indices
        df = df.set_index("LIWC category and content examples")
        columns = [("ttest", c[0]) if len(c) == 1 else tuple(c) for c in df.columns.str.split()]
        df.columns = pd.MultiIndex.from_tuples(columns)
        df = (df
            .rename(columns={"Normative": "HVdC"})
            .rename_axis(index="Category", columns=[None, "Parameter"])
            .sort_index(axis="index").sort_index(axis="columns")
        )
    return df


def load_cariola2014(table):
    """
    Table 1: Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of Low and High Barrier Personalities in Narratives of Everyday Memories
    Table 2: Univariate Results of Body Boundary Imagery and LIWC Linguistic Variables of Low and High Barrier Personalities in Narratives of Dream Memories
    """
    fp = _compile_table_filepath("cariola2014", table)
    if table != "manual":
        df = (
            pd.read_table(fp, index_col=0, skiprows=1, header=[0, 1])
            .rename(columns={"Unnamed: 2_level_0": "Low", "Unnamed: 4_level_0": "High"})
            .rename_axis(index="Category", columns=["BarrierPersonality", "Parameter"])
            .astype(float)
            .sort_index(axis="index").sort_index(axis="columns")
        )
        # df.index = df.index.astype("str").categorical
    return df


def load_bulkeley2018(table):
    fp = _compile_table_filepath("bulkeley2018", table)
    if table != "manual":
        df = (
            pd.read_table(fp)
            .astype({
                "Category name": "str",
                "SDDb baselines (male and female)": "float",
                "SD": "float"
            })
            .rename(columns={
                "Category name": "Category",
                "SDDb baselines (male and female)": "Mean",
            })
        )
        df["Category"] = pd.Categorical(df["Category"], ordered=False)
        df = df.set_index("Category").sort_index(axis="index").sort_index(axis="columns")
    return df


def load_cariola2010(table):
    fp = _compile_table_filepath("cariola2010", table)
    if table != "manual":
        df.index = pd.CategoricalIndex(df.index.astype(str), ordered=False)
        df = (
            pd.read_table(fp, index)
            .rename(columns={"Linguistic processes": "Category"})
            .astype({
                "Category": "str",
                "Maximum": "float",
                "Minimum": "float",
                "Mean": "float",
                "SD": "float",
            })
        )
        df["Category"] = (
            df["Category"].str.lower()
            .replace(
                {
                    "Linguistic processes": {
                        "word count": "WC",
                        "1st person singular pronouns": "i",
                        "3rd person singular pronouns": "we",
                        "past tense": "past",
                        "present tense": "present",
                        "adverbs": "adverb",
                        "quantifiers": "quant",
                        "positive emotions": "posemo",
                        "negative emotions": "negemo",
                        "causation": "caus",
                        "Tentativeness": "tentat",
                        "certainty": "certain",
                        "inclusion": "inclu",
                    }
                }
            )
        )
        df = df.set_index("Category").sort_index(axis="index").sort_index(axis="columns")
    return df


def load_hawkins2017(table):
    fp = _compile_table_filepath("hawkins2017", table)
    if table != "manual":
        df = (
            pd.read_table(fp, index_col=0, header=[0, 1])
            .rename(columns=
                {
                    "Unnamed: 2_level_0": "2007 norms",
                    "Unnamed: 4_level_0": "Study 1 Time 1",
                    "Unnamed: 6_level_0": "Study 1 Time 2",
                    "Unnamed: 8_level_0": "Amazon Mechanical Turk",
                    "Unnamed: 10_level_0": "TOWER",
                    "Unnamed: 12_level_0": "Ave. recent dream",
                }
            )
            .rename(columns=
                {
                    "Amazon Mechanical Turk": "MTurk",
                    "Ave. recent dream": "AvgRecent",
                }
            )
            .astype(float)
            .rename_axis(index="Category", columns=["Study", "Parameter"])
        )
        df.index = pd.CategoricalIndex(df.index.astype(str), ordered=False)
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df


def load_mariani2023(table):
    fp = _compile_table_filepath("mariani2023", table)
    if table != "manual":
        df = pd.read_table(fp, index_col=0)
        df = df.astype(float)
        columns = df.columns.str[8:].str.rstrip(")").str.split(" \\(")
        df.columns = pd.MultiIndex.from_tuples([tuple(c) for c in columns])
        df = df.rename_axis(index="Category", columns=["Cluster", "Parameter"])
        df.index = (
            df.index.str.lower()
            .to_series()
            .replace({"To be": "be"})
        )
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df


def load_mcnamara2015(table):
    fp = _compile_table_filepath("mcnamara2015", table)
    if table != "manual":
        multiheader_rows = [0, 1]
        n_header_rows = len(multiheader_rows)
        df = pd.read_table(fp, index_col=0, header=None)
        header_rows = df.iloc[:n_header_rows]
        header_filled = header_rows.ffill(axis=1)
        multi_index = pd.MultiIndex.from_arrays(header_filled.values)
        df = df.rename_axis(index=df.index[n_header_rows-1])
        df = df.iloc[n_header_rows:]
        df.columns = multi_index
        df = df.rename_axis(columns=["Sample", "Parameter"])
        # df = df.rename(columns={"Mean": "M"})
        df.index = pd.CategoricalIndex(df.index.astype(str), ordered=False)
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df

def load_meador2022(table):
    fp = _compile_table_filepath("meador2022", table)
    if table != "manual":
        df = pd.read_table(fp)
        df["Category"] = pd.Categorical(df["Category"].astype(str), ordered=False)
        df = df.set_index("Category")
        df = df.astype(float)
        df = df.rename(columns={"Nightmare (Mean)": "Mean", "Nightmare (SD)": "SD"})
        df = df.sort_index(axis=0).sort_index(axis=1)
    return df

def load_mota2020(table):
    fp = _compile_table_filepath("mota2020", table)
    df = pd.read_table(fp, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_frame(
        (
            df.columns
            # .rename(["Analysis", "Parameter"])
            .to_frame().reset_index(drop=True)
            .replace({":": pd.NA}, regex=True)
            .ffill(axis=0)
            .replace(" Analysis", "", regex=True)
            .astype(str)
        )
    )
    df = df.rename_axis(index="Participant_ID", columns=["Analysis", "Parameter"])
    df = df.astype(float)
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df


def load_niederhoffer2017(table):
    fp = _compile_table_filepath("niederhoffer2017", table)
    df = pd.read_table(fp, index_col=0)
    df = df.astype(float)
    df.columns = pd.MultiIndex.from_tuples(tuple(c) for c in df.columns.str.rstrip(")").str.split(" \\("))
    df = df.rename_axis(columns=["Sample", "Parameter"])
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df


def load_zheng2023(table):
    fp = _compile_table_filepath("zheng2023", table)
    df = pd.read_table(fp, index_col=0, dtype={"Categorical": "str", "Dream (M)": "float", "Wake (M)": "float"})
    df.columns = df.columns.str.split().str[0]
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df

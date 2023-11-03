"""
Make the abbreviation map based on CCF ontology.
"""
import pandas as pd
from myle import nav

df = pd.read_excel("/root/capsule/data/allen_ccf_v3/allen_ccf_v3.xlsx", header=1)
structure = df["full structure name"]
abbreviation = df["abbreviation"]

areas_dict = {}
for struct, abbr in zip(structure, abbreviation):
    areas_dict[struct] = abbr
nav.pklsave(areas_dict, "areas_dict.pkl")

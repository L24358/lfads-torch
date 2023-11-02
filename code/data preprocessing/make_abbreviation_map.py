"""
Make the abbreviation map based on CCF nomanclature.
"""
import pandas as pd

df = pd.read_excel("/root/capsule/data/allen_ccf_v3/allen_ccf_v3.xlsx", header=1)
structure = df["full structure name"]
abbreviation = df["abbreviation"]

areas_dict = {}
for struct, abbr in zip(structure, abbreviation):
    areas_dict[struct] = abbr

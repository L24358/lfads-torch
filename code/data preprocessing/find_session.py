"""
Find appropriate subject/session given required brain areas and inclusion criterion.

Args:
    - required_areas (list): brain areas that need to be included
    - criterions (callable): determines whether a brain area meets the criterions set
"""
import pandas as pd
from myle import nav

class Brain:
    def __init__(self, df):
        self.df = df
        self._build()
        self._make_abbreviation_map()
        
    def assign_interest_group(self, group, data_type="id"):
        if data_type == "id": pass
        elif data_type == "abbrev": group = [self.id_dict[member] for member in group]
        elif data_type == "name": group = [self.id_dict[self.abbrev_dict[member]] for member in group]
        self.interest_group = group
        
    def assign_uniform_level(self, level):
        flag = False
        for id in self.interest_group:
            member = self.brain_areas[id]
            if member.level >= level: member.level_up(level)
            else: flag = True
            
        if flag: print(f"Some members are higher than the assigned level, level {level}, and cannot be converted.")
        return flag
        
    def _build(self):
        self.brain_areas = {}
        for idx, row in self.df.iterrows():
            self.brain_areas[row["structure ID"]] = BrainArea(
                row["full structure name"],
                row["structure ID"],
                row["structure_id_path"],
                row["abbreviation"],
                row["depth in tree"])
    
    def _make_abbreviation_map(self):
        self.abbrev_dict = {}
        self.id_dict = {}
        for struct, abbr, id in zip(self.df["full structure name"], self.df["abbreviation"], self.df["structure ID"]):
            self.abbrev_dict[struct] = abbr
            self.id_dict[abbr] = id

class BrainArea:
    def __init__(self, name, id, path, abbrev, depth):
        self.name = name
        self.id = id
        self.path = list(filter(None, path.split('/')))
        self.abbrev = abbrev
        self.level = depth
        
        self.current_id = id
        self.current_level = depth
        
    def level_up(self, val, mode="abs"):
        if mode == "abs":
            self.current_level = val
        elif mode == "rel":
            self.current_level += val
        else:
            raise ValueError
            
        self.current_id = self.path[self.current_level]
        
if __name__ == "__main__":
    
    # USER DEFINED VARIABLES
    required_areas = ["Thalamus", "Somatomotor areas", "Secondary motor area", "Primary motor area"]
    level = 4
    
    # Build ccf brain
    ccf_df = pd.read_excel("/root/capsule/data/allen_ccf_v3/allen_ccf_v3.xlsx", header=1)
    brain = Brain(ccf_df)
    brain.assign_interest_group(required_areas, data_type="name")
    brain.assign_uniform_level(level)
    
    # process summary tables
    files = NotImplemented
    summary_df = pd.concat([pd.read_csv(fname) for fname in files])
    summary_df = summary_df.drop(["firing rate std", "firing rate per neuron"])
    g = summary_df.group_by(["subject", "session"])
    
    for group_name, group in g.items():
        pass
    
    
        
    
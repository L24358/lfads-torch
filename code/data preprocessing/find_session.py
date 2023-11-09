"""
Find appropriate subject/session given required brain areas and inclusion criterion.

Use required_areas = ["Primary somatosensory area, lower limb, layer 4"] for debugging.
"""
import os
import numpy as np
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
        elif data_type == "name": group = [self.id_dict[member] for member in group]
        self.interest_group = group
        
    def assign_uniform_level(self, level):
        flag = False
        for id in self.interest_group:
            member = self.brain_areas[id]
            if member.level >= level: member.level_up(level)
            else: flag = True
            
        # update into unique ids
        self.interest_group = list(set([self.brain_areas[id].current_id for id in self.interest_group]))
            
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
            
    def is_available(self, target_id, available_ids):
        target = self.brain_areas[target_id]
        av_idxs = []
        
        for av_idx, av_id in enumerate(available_ids):
            candidate = self.brain_areas[av_id]
            if target_id in candidate.path[target.current_level:]: av_idxs.append(av_idx)
        if len(av_idxs) > 0: return True, av_idxs
        else: return False, None
    
    def get_name(self, ids):
        return "; ".join([self.rev_id_dict[id] + f" ({self.abbrev_dict[self.rev_id_dict[id]]})" for id in ids])
    
    def _make_abbreviation_map(self):
        self.abbrev_dict = {}
        self.id_dict = {}
        self.rev_id_dict = {}
        for struct, abbr, id in zip(self.df["full structure name"], self.df["abbreviation"], self.df["structure ID"]):
            self.abbrev_dict[struct] = abbr
            self.id_dict[struct] = id
            self.rev_id_dict[id] = struct

class BrainArea:
    def __init__(self, name, id, path, abbrev, depth):
        self.name = name
        self.id = id
        self.path = [int(item) for item in filter(None, path.split('/'))]
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
    
def num_neurons_criterion(num_neurons, thre=10): return num_neurons > thre

def frate_criterion(frate, thre=0.01): return frate > thre
        
if __name__ == "__main__":
    
    # USER DEFINED VARIABLES
    required_areas = ["Ventral medial nucleus of the thalamus", "Ventral anterior-lateral complex of the thalamus", "Secondary motor area, layer 5", "Secondary motor area, layer 2/3", "Somatomotor areas", "Midbrain"] # "Pons"
    uniform_level = False
    level = 4 # only required when uniform_level = True
    
    # Build ccf brain
    summary_data_path = "/root/capsule/data/chen2023_summary_statistics/"
    ccf_df = pd.read_excel(summary_data_path + "allen_ccf_v3.xlsx", header=1)
    brain = Brain(ccf_df)
    brain.assign_interest_group(required_areas, data_type="name")
    if uniform_level: brain.assign_uniform_level(level)
    print("Interested areas: ", brain.get_name(brain.interest_group), "\n")
    
    # process summary tables
    files = [os.path.join(summary_data_path, f) for f in os.listdir(summary_data_path) if "summary" in f]
    summary_df = pd.concat([pd.read_csv(fname) for fname in files])
    summary_df = summary_df.drop(["firing rate std", "firing rate per neuron"], axis=1)
    g = summary_df.groupby(["subject", "session"])
    
    for group_name, group in g:
        available_areas = np.array([brain.id_dict[row["area"]] for _, row in group.iterrows()])
        
        # if targets are in available areas
        areas_in, areas_pass_criterion = [], []
        num_neurons_list = []
        
        for target_id in brain.interest_group:
            is_in, sub_area_idxs = brain.is_available(target_id, available_areas) 
            if is_in:
                areas_in.append(target_id)
                
                # if available areas meet numbers and firing rate criterion
                num_neurons, frate = [], []
                for sub_area_idx in sub_area_idxs:
                    row = group.iloc[sub_area_idx]
                    num_neurons.append(row["# neurons"])
                    frate.append(row["firing rate mean"])
                frate = np.dot(np.array(num_neurons), np.array(frate)) / sum(num_neurons)
                num_neurons = sum(num_neurons)
                num_neurons_list.append(num_neurons)
                    
                if num_neurons_criterion(num_neurons) and frate_criterion(frate):
                    areas_pass_criterion.append(target_id)
    
        if len(areas_pass_criterion) == len(brain.interest_group):
            print("Group ", group_name, " satisfied all criteria.")
            print("The number of neurons are: ", num_neurons_list)
        elif len(areas_in) == len(brain.interest_group):
            print("Group ", group_name, " satisfied the 'available' criterion.")
        
    
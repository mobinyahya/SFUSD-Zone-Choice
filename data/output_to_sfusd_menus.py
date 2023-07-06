import numpy as np
import pandas as pd
import json


class MakeMenu:
    def __init__(self, name):
        self.name = name
        df = pd.read_csv(f"assignment_plans/large_market_plans/{name}/menus.csv")
        self.ref = pd.read_csv("klm_raw_inputs/program_idxs.csv")
        self.idx2id = dict(zip(self.ref.program_idx, self.ref.program_id))
        ref2 = pd.read_csv("klm_raw_inputs/typecodes.csv", index_col=0)[
            ["census_blockgroup", "lang_match"]
        ]
        df = df.merge(ref2, how="left", left_on="Type", right_on="type_code")
        tmp = df.drop(columns=["159"])
        for col in range(159):
            if str(col) not in df.columns:
                print(col, "not found")
                continue
            tmp.loc[:, str(col)] = tmp[str(col)] * tmp.Prob
        self.type_avg = tmp.groupby(["Type", "lang_match", "census_blockgroup"], as_index=False).sum()
        self.type_avg["typecode"] = self.type_avg.apply(lambda x: f"{x.lang_match}-{x.census_blockgroup}", axis=1)

        self.no_sib, self.siblings, self.st = self.load_data()
        self.stno2idx = dict(zip(self.st.studentno, self.st.index))

    def build_menus(self):
        sibling_menus = self.sibling_menus()
        rest_menus = self.non_sibling_menus()
        menus = {**sibling_menus, **rest_menus}
        # give students with missing census block groups access to all schools
        for studentno in list(self.st.loc[self.st.census_blockgroup.isna()].studentno):
            menus[studentno] = list(self.ref.program_id)

        save_path = f"/Users/katherinementzer/Documents/sfusd/local_runs/Zones/peng_menu_{self.name}.json"
        with open(save_path, "w") as f:
            json.dump(menus, f)
        print(f"Menu saved to {save_path}")

    @staticmethod
    def load_data():
        st = pd.read_csv("~/SFUSD/Data/Cleaned/drop_optout_1819.csv", low_memory=False)
        st = st.loc[st.grade == "KG"]
        siblings = pd.read_csv("klm_raw_inputs/preassigned_siblings.csv")
        no_sib = pd.merge(st, siblings, on="studentno", how="outer", indicator=True)
        no_sib = no_sib[no_sib['_merge'] == 'left_only']
        no_sib = no_sib.dropna(subset=["census_blockgroup"])
        no_sib.census_blockgroup = no_sib.census_blockgroup.astype(int)
        no_sib.loc[:, "lang_match"] = no_sib.homelang_desc.apply(
            lambda x: "cantonese"
            if x == "CC-Chinese Cantonese"
            else ("spanish" if x == "SP-Spanish" else "no_match")
        )
        no_sib["typecode"] = no_sib.apply(lambda x: f"{x.lang_match}-{x.census_blockgroup}", axis=1)

        siblings = siblings.merge(st[["studentno", "sibling"]], how="left")
        st = st.reset_index(drop=True)
        return no_sib, siblings, st

    def sibling_menus(self):
        # self.siblings.loc[:, "sibling"] = self.siblings.sibling.apply(lambda x: eval(x)[0])
        # self.siblings.loc[:, "program_id"] = [f"{x}-GE-KG" for x in self.siblings.sibling]
        # sibling_menus = {row.studentno: [row.program_id] for _, row in self.siblings.iterrows()}
        # print(sibling_menus)
        # exit()
        with open("/Users/katherinementzer/SFUSD/sibling_menus.json", "r") as f:
            sibling_menus = json.load(f)
        return sibling_menus

    def non_sibling_menus(self):
        self.type_avg["typecode"] = self.type_avg.apply(lambda x: f"{x.lang_match}-{x.census_blockgroup}", axis=1)
        rest_menus = {
            row.typecode: [self.idx2id[x] for x in range(159) if str(x) in self.type_avg.columns and row[str(x)] > 0]
            for _, row in self.type_avg.iterrows()
        }
        rest_menus = {row.studentno: rest_menus[row.typecode] for _, row in self.no_sib.iterrows()}
        return rest_menus

    def make_priority_boost_matrix(self):
        boost_matrix = np.zeros((4772, 159))
        self.no_sib.loc[:, "priority_idx"] = self.no_sib.studentno.apply(lambda x: self.stno2idx[x])
        type_avg = self.type_avg.set_index("typecode")
        for idx, row in self.no_sib.iterrows():
            for j in range(159):
                if str(j) not in type_avg.columns:
                    continue
                boost_matrix[row.priority_idx, j] = type_avg.loc[row.typecode, str(j)]
        save_name = f"/Users/katherinementzer/Documents/sfusd/local_runs/Data/Precomputed/peng_boost_matrix_{self.name}.npy"
        np.save(save_name, boost_matrix)
        print(f"Boost matrix saved to {save_name}")


if __name__ == "__main__":
    mm = MakeMenu("dist10_maxfrl1_alpha0.5_card0_umodelavg")
    mm.build_menus()
    mm.make_priority_boost_matrix()


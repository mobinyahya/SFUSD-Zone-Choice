from typing import List, Dict

import gurobipy as guro

sizeActive = False
feasibilityActive = True

distActive = True
cardActive = True
capacityActive = True
frlActive = True


class Constraints:
    feasibilityConstr: Dict[int, guro.Constr]

    distConstr: guro.Constr
    capacityConstr: List[guro.Constr]
    equityConstr: Dict[int, guro.Constr]
    sizeConstr: guro.Constr
    cardConstr: guro.Constr
    frl_constr: Dict[int, guro.Constr]

    def __iter__(self):
        yield self.distConstr
        yield self.capacityConstr
        yield self.feasibilityConstr
        yield self.equityConstr
        yield self.sizeConstr
        yield self.cardConstr
        yield self.frl_constr


class Variables:
    feasibilityVar: Dict[int, guro.Var]

    distVar: guro.Var
    capacityVar: List[guro.Var]
    cardVar: guro.Var
    frlVar: Dict[int, guro.Var]

    def __iter__(self):
        yield self.distVar
        yield self.capacityVar
        yield self.feasibilityVar
        yield self.cardVar
        yield self.frlVar

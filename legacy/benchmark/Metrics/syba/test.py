#from rdkit import Chem
from syba.syba import SybaClassifier

from rdkit.Chem import AllChem as Chem
syba = SybaClassifier()
syba.fitDefaultScore()





print(syba.predict("CCO"))



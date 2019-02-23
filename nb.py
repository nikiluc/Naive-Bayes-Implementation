import TenCrossValidation as tcv
from Preprocessor import preprocess
import sys


file = sys.argv[1]

data = preprocess(file)

result = tcv.validation(data)




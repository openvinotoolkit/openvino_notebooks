from sklearnex import patch_sklearn
from sklearnex import unpatch_sklearn
patch_sklearn()
print("To disable Intel(R) Extension for Scikit-learn*, you can run: unpatch_sklearn()")
how to use function

linear regression have 2 optional arguments,
alpha and epocs/iterations

use by linearRegression(alpha=0.1,epocs=1000) //default values

for changing evaluation schema

call fit ( x, y ,option=t) : option argument is optional 
t= "RMSE" for RMSE
"MAE" for MAE
and default is RMSE


logistic regression have 2 optional arguments,
alpha and epocs/iterations

use by logisticRegression(alpha=0.1,epocs=3000) //default values

for changing evaluation schema

call fit ( x, y ,option=t) : option argument is optional 
t= "SGD" for scholastic
"BGD" or batch for default

### no k fold etc implemented, though fcntion present.

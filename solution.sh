# delete previous solution

SOLUTION_NAME='solution_unet.pyz'

rm $SOLUTION_NAME

python3.6 -m zipapp solution_unet -p='/usr/bin/env python3.6'

echo "solution is created"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$SCRIPT_DIR/..

rm $BASE_DIR/results/glued_variables.json
echo {}> $BASE_DIR/results/glued_variables.json

rm $BASE_DIR/results/glued_variables.txt
touch $BASE_DIR/results/glued_variables.txt
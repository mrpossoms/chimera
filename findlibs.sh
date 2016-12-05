#!/bin/sh

cat /dev/null > findings.mk

function find_lib_dir {
	LIB=$1
	LIB_PATH=$(find /opt /usr/local -name $LIB -print -quit)
	echo ${LIB_PATH/\/$LIB/}
}

MAKE_LIB=
MAKE_INC=

echo "Finding libraries..."

for DEPENDENCY in $(cat depends); do
	DEPENDENCY=$(echo $DEPENDENCY | tr -d '\n');
	echo $DEPENDENCY;
	LIB_DIR=$(find_lib_dir $DEPENDENCY)
	INC_DIR=${LIB_DIR/lib/include}
	MAKE_LIB+=$(printf " -L%s" $LIB_DIR);
	MAKE_INC+=$(printf " -I%s" $INC_DIR)
done

printf "LIB_PATHS+=%s\n" "$MAKE_LIB" >> findings.mk
printf "INC_PATHS+=%s" "$MAKE_INC" >> findings.mk

echo "Done!"

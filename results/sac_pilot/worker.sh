python -s -m sim.train > train.log 2>&1 &
child=$!
printf "%s\n" "$child" > train.pid
wait "$child"
result=$?
printf "%s\n" "$result" > train.exit
exit "$result"

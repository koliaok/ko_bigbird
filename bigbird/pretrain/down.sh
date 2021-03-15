
for i in 1 2
do
    var1=$(ps -ef | grep 'run_pretraining\.py')
    echo process info : ${var1}
    second1=$(echo ${var1} | cut -d " " -f2)
    if [ -n "${second1}" ]
    then
        result1=$(kill -9 ${second1})
        echo process is killed.
    else
        echo running process not found.
    fi
done

# Run E3SM tests repeatedly.

pass=0
fail=0

while true; do
    rm -f TestStatus
    ./case.submit
    while true; do
        if [ -f TestStatus ]; then
            ln=`grep COMPARE_base_rest TestStatus`
            echo $ln
            if [[ $ln =~ PASS ]]; then
                pass=$(( $pass + 1 ))
            elif [[ $ln =~ FAIL ]]; then
                fail=$(( $fail + 1 ))
            else
                echo "wait 2"
                sleep 120
                continue
            fi
            printf "pass %d fail %d\n" $pass $fail
            break
        else
            echo "wait 1"
            sleep 120
        fi
    done
    if [[ $fail -gt 20 ]]; then
        break
    fi
    if [[ $pass -gt 20 ]]; then
        break
    fi
done

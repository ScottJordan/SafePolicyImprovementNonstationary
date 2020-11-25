for d in */ ; do
        cd $d
        cat 1_*.csv >> allact1.csv
        cat 2_*.csv >> allact2.csv
        cat 3_*.csv >> allact3.csv
        cat 4_*.csv >> allact4.csv
        cat 5_*.csv >> allact5.csv
        cd ..
done

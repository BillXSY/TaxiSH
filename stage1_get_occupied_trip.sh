#!/bin/bash

# -----------------------------------------------------------------
# 过滤和处理数据：只保留载客的行程，并对每个出租车的行程id进行标记
# -----------------------------------------------------------------

input_file=""
output_file=""


date
echo "Filtering and processing data..."

> "$output_file"

awk 'BEGIN{OFS=" "; globalTripID=0} {
    lon=$4; lat=$5; ifPassenger=$6; uid=$3;

    prevState[uid] = prevState[uid] ? prevState[uid] : 0;
    # tripCnt[uid] = tripCnt[uid] ? tripCnt[uid] : 0;

    if (ifPassenger > 0 && prevState[uid] == 0) {
        # tripCnt[uid]++;
        globalTripID++;
    }

    if (ifPassenger > 0 && lon != -1.0 && lat != -1.0 && uid > 0) {
        # print $0 " " tripCnt[uid] " " globalTripID;
        print $0 " " globalTripID;
    }

    prevState[uid] = ifPassenger;
    }'  "$input_file" >> "$output_file"

date
echo "Data filtering and processing completed. Output saved to: $output_file"

# -----------------------------------------------------------------
#!/bin/bash

# -----------------------------------------------------------------
# 提取OD对，并保留起止时间和日期、OD haversine距离、行程总距离、记录数
# -----------------------------------------------------------------

date
echo "提取OD对，并保留起止时间和日期、OD haversine距离、行程总距离、记录数"

input_file=""
output_file=""

awk -F ',' '
function haversine(lon1, lat1, lon2, lat2, dlon, dlat, a, c) {
    # 将十进制度数转换为弧度
    lon1 = lon1 * 0.01745329252  # π/180 ≈ 0.01745329252
    lat1 = lat1 * 0.01745329252
    lon2 = lon2 * 0.01745329252
    lat2 = lat2 * 0.01745329252
    
    # 计算经纬度差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine公式核心计算
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return 6371 * c  # 返回距离（公里）
}
BEGIN{
    OFS=",";
    prevkey="";
    
}
{
    uid=$4; tid=$11;
    key=uid","tid;

    if (key != prevkey) {
        if (prevkey != "") {
            # 结算并打印上一个 Trip 的数据
            od_distance = haversine(s_lon, s_lat, p_lon, p_lat);
            trip_time = p_time_s - s_time_s;
            print p_uid, p_tid, s_lon, s_lat, p_lon, p_lat, s_date, s_time, p_date, p_time, od_distance, total_dist, trip_time, count;
        }
        # 初始化当前新 Trip 的起始状态
        prevkey = key
        p_uid = uid
        p_tid = tid
        
        s_lon = $5; s_lat = $6;
        s_date = $1; s_time = $2;
        s_time_s = $3; # 直接使用第3列的时间戳！
        
        total_dist = 0
        count = 1
        
        # 记录前一个点的坐标，用于下次计算累加距离
        p_lon = $5; p_lat = $6;
        p_date = $1; p_time = $2;
        p_time_s = $3;
        
    } else {
        # 属于同一个 Trip，累加状态
        total_dist += haversine(p_lon, p_lat, $5, $6);
        count++;
        
        # 更新前一个点的状态为当前行，供下一次迭代使用
        p_lon = $5; p_lat = $6;
        p_date = $1; p_time = $2;
        p_time_s = $3;
    }
} 
END {
    if (prevkey != "") {
        # 输出最后一个key的结果
        od_distance = haversine(s_lon, s_lat, p_lon, p_lat);
        trip_time = p_time_s - s_time_s;
        print p_uid, p_tid, s_lon, s_lat, p_lon, p_lat, s_date, s_time, p_date, p_time, od_distance, total_dist, trip_time, count;
    }
}' "$input_file" > "$output_file"

date
echo "OD对提取完成"
echo "排序输出文件..."
sort -k1,1 -k2,2n -u --parallel=64 -S 500G "$output_file" -o "$output_file"

# -----------------------------------------------------------------


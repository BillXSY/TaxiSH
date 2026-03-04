import pandas as pd
import numpy as np
import geopandas as gpd

minLon = 120.828879
maxLat = 31.879505
resolution = 0.0083
dT = 1800  # 30 minutes
start_utime = 1404172800 - 8 * 3600 # 2014-07-01 00:00:00

input_file = ''
output_file = ''
shp_path = ""


od_df = pd.read_csv(input_file, header=None, sep=',')

od_df.columns = ['userID', 'trajID', 'O_lon', 'O_lat', 'D_lon', 'D_lat', 'O_date', 'O_time', 'D_date', 'D_time', 'od_dist', "trip_dist", 'trip_time', "rec_cnt"]

print(len(od_df))


# 2. 初步空间及记录数过滤
od_df = od_df.query("rec_cnt >= 5 and 120.82 < O_lon < 122 and 30.6 < O_lat < 31.88 and 120.82 < D_lon < 122 and 30.6 < D_lat < 31.88")
print(len(od_df))


# 3. 高效的时间处理 (先一次性转为 Datetime)
od_df['O_datetime'] = pd.to_datetime(od_df['O_date'] + ' ' + od_df['O_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
od_df['D_datetime'] = pd.to_datetime(od_df['D_date'] + ' ' + od_df['D_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# 清除解析失败的异常行
od_df = od_df.dropna(subset=['O_datetime', 'D_datetime'])

# 4. 先按时间过滤 (减少后续计算量)
od_df = od_df.query("O_datetime.dt.year == 2014 and 7 <= O_datetime.dt.month <= 10 and D_datetime.dt.year == 2014 and 7 <= D_datetime.dt.month <= 10")
print(len(od_df))


# 5. 修复时区问题并计算 Unixtime
# 假设源数据是北京时间，先 localized 到 Asia/Shanghai，再转为 UTC 时间戳
od_df['O_unixtime'] = od_df['O_datetime'].dt.tz_localize('Asia/Shanghai').astype("int64") // 10**9
od_df['D_unixtime'] = od_df['D_datetime'].dt.tz_localize('Asia/Shanghai').astype("int64") // 10**9

# 6. 先过滤异常的时间和距离，避免计算速度时除以零
od_df = od_df.query("10 <= trip_time < 86400 and 0.1 <= od_dist <= 200 and 0.1 <= trip_dist <= 200")
print(len(od_df))

# 7. 特征工程
od_df['od_speed'] = od_df['od_dist'] / (od_df['trip_time'] / 3600)  # km/h
od_df['trip_speed'] = od_df['trip_dist'] / (od_df['trip_time'] / 3600)  # km/h

# 再次过滤速度异常值
od_df = od_df.query("1 <= trip_speed <= 150")
print(len(od_df))


# 网格与时间切片
# 注意: np.floor 更严谨，因为负数的 floor division `//` 在浮点数下可能存在极小精度误差
od_df['O_x'] = np.floor((od_df['O_lon'] - minLon) / resolution).astype(int)
od_df['O_y'] = np.floor((maxLat - od_df['O_lat']) / resolution).astype(int) # 改为 maxLat 减 lat 避免负数除法
od_df['D_x'] = np.floor((od_df['D_lon'] - minLon) / resolution).astype(int)
od_df['D_y'] = np.floor((maxLat - od_df['D_lat']) / resolution).astype(int)
od_df['O_t'] = ((od_df['O_unixtime'] - start_utime) // dT).astype(int)
od_df['D_t'] = ((od_df['D_unixtime'] - start_utime) // dT).astype(int)

# 8. 空间匹配 (极简内存版)
gdf = gpd.read_file(shp_path).set_crs("+proj=lcc +lat_1=30.704202 +lat_2=31.866952 +lat_0=31.2355  +lon_0=121.467 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1", allow_override=True)
gdf = gdf.to_crs(epsg=4326)

# [关键优化]: 只用坐标和索引创建 GDF，极大节省内存
gdf_O = gpd.GeoDataFrame(index=od_df.index, geometry=gpd.points_from_xy(od_df.O_lon, od_df.O_lat), crs=4326)
gdf_D = gpd.GeoDataFrame(index=od_df.index, geometry=gpd.points_from_xy(od_df.D_lon, od_df.D_lat), crs=4326)

# [关键优化]: 只取用得到的 OBJECTID 字段和 geometry 去 join
o_join = gpd.sjoin(gdf_O, gdf[['OBJECTID', 'geometry']], how="left", predicate="intersects")
d_join = gpd.sjoin(gdf_D, gdf[['OBJECTID', 'geometry']], how="left", predicate="intersects")

# [关键修复]: 去重，防止正好压在多边形边界上的点生成重复行导致赋值报错
o_join = o_join[~o_join.index.duplicated(keep='first')]
d_join = d_join[~d_join.index.duplicated(keep='first')]

od_df["O_belong_id"] = o_join["OBJECTID"]
od_df["D_belong_id"] = d_join["OBJECTID"]

# 提取在面内的记录 (剔除 OBJECTID 为 NaN 的行)
od_df_clean = od_df.dropna(subset=["O_belong_id", "D_belong_id"])
print(len(od_df))


# 9. 输出
# 删除过程中产生的临时日期列 (可选，可减小输出体积)
od_df_clean = od_df_clean.drop(columns=['O_datetime', 'D_datetime'])
od_df_clean.to_csv(output_file, index=False, header=False, sep=' ')
print(f"处理完成，最终有效数据量: {len(od_df_clean)}")

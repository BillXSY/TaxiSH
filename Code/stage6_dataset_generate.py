import pandas as pd
import numpy as np
import os
import time

def main():
    stt = time.time()
    
    # 1. 定义路径与列名
    input_file = ''
    output_dir = ''
    
    # 如果输出文件夹不存在，则创建
    os.makedirs(output_dir, exist_ok=True)
    
    columns = [
        'userID', 'trajID', 'O_lon', 'O_lat', 'D_lon', 'D_lat', 
        'O_date', 'O_time', 'D_date', 'D_time', 'od_dist', "trip_dist", 
        'trip_time', "rec_cnt", "O_unixtime", "D_unixtime", "od_speed",
        "trip_speed", 'O_x', 'O_y', 'D_x', 'D_y', 'O_t', 'D_t', 
        'O_belong_id', 'D_belong_id'
    ]

    print(f"[{time.strftime('%H:%M:%S')}] 正在读取清洗后的底表数据...")
    df = pd.read_csv(input_file, sep=' ', header=None, names=columns)
    
    # 使用 O_x 和 O_y 拼接生成 O_Grid_ID，D_x 和 D_y 拼接生成 D_Grid_ID
    df['O_Grid_ID'] = df['O_x'].astype(str) + '_' + df['O_y'].astype(str)
    df['D_Grid_ID'] = df['D_x'].astype(str) + '_' + df['D_y'].astype(str)
    
    print(f"[{time.strftime('%H:%M:%S')}] 数据加载完成，共 {len(df)} 条记录。开始生成标准数据集...")

    # ==========================================
    # 数据集 1: Trip-Level OD Dataset
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 正在生成 1. Trip-Level OD Dataset...")
    trip_level_df = df[[
        'userID', 'O_t', 'O_Grid_ID', 'D_t', 'D_Grid_ID', 
        'trip_dist', 'trip_time', 'trip_speed'
    ]].copy()
    
    trip_level_df.columns = [
        'Taxi_ID', 'O_Time_Index', 'O_Grid_ID', 'D_Time_Index', 'D_Grid_ID', 
        'Dist_km', 'Dur_s', 'Speed_kmh'
    ]
    
    trip_level_df['Dist_km'] = trip_level_df['Dist_km'].round(2)
    trip_level_df['Speed_kmh'] = trip_level_df['Speed_kmh'].round(2)
    
    trip_level_file = os.path.join(output_dir, 'TaxiSH_Trip_Level_OD.csv')
    trip_level_df.to_csv(trip_level_file, index=False)


    # ==========================================
    # 数据集 2: Spatiotemporal OD Flow Dataset
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 正在生成 2. OD Flow Dataset...")
    od_flow_df = df.groupby(['O_t', 'O_Grid_ID', 'D_Grid_ID'], as_index=False).agg(
        Trip_Volume=('userID', 'size'),
        Avg_Dist_km=('trip_dist', 'mean'),
        Avg_Dur_s=('trip_time', 'mean'),
        Avg_Speed_kmh=('trip_speed', 'mean')
    )
    
    od_flow_df.rename(columns={'O_t': 'Time_Index'}, inplace=True)
    
    od_flow_df['Avg_Dist_km'] = od_flow_df['Avg_Dist_km'].round(2)
    od_flow_df['Avg_Dur_s'] = od_flow_df['Avg_Dur_s'].round(0).astype(int)
    od_flow_df['Avg_Speed_kmh'] = od_flow_df['Avg_Speed_kmh'].round(2)
    
    od_flow_df.sort_values(['Time_Index', 'O_Grid_ID', 'D_Grid_ID'], inplace=True)
    
    od_flow_file = os.path.join(output_dir, 'TaxiSH_OD_Flow.csv')
    od_flow_df.to_csv(od_flow_file, index=False)


    # ==========================================
    # 数据集 3: In/Out Flow Dataset
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 正在生成 3. In/Out Flow Dataset...")
    
    outflow = df.groupby(['O_t', 'O_Grid_ID']).size().reset_index(name='Outflow')
    outflow.rename(columns={'O_t': 'Time_Index', 'O_Grid_ID': 'Grid_ID'}, inplace=True)
    
    inflow = df.groupby(['D_t', 'D_Grid_ID']).size().reset_index(name='Inflow')
    inflow.rename(columns={'D_t': 'Time_Index', 'D_Grid_ID': 'Grid_ID'}, inplace=True)
    
    inout_flow_df = pd.merge(outflow, inflow, on=['Time_Index', 'Grid_ID'], how='outer')
    inout_flow_df.fillna(0, inplace=True)
    inout_flow_df['Outflow'] = inout_flow_df['Outflow'].astype(int)
    inout_flow_df['Inflow'] = inout_flow_df['Inflow'].astype(int)
    inout_flow_df['Total_Flow'] = inout_flow_df['Inflow'] + inout_flow_df['Outflow']
    
    inout_flow_df.sort_values(['Time_Index', 'Grid_ID'], inplace=True)
    
    inout_flow_file = os.path.join(output_dir, 'TaxiSH_InOut_Flow.csv')
    inout_flow_df.to_csv(inout_flow_file, index=False)

    print(f"[{time.strftime('%H:%M:%S')}] 全部数据集生成完毕！总耗时: {time.time() - stt:.2f} 秒。")
    print(f"文件已保存至: {output_dir}")

if __name__ == '__main__':
    main()

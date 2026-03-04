import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
import os
import time
import glob
import shutil
from tqdm import tqdm

# --- 1. 辅助函数：向量化 Haversine (直接复用之前的高性能版本) ---
def haversine_vectorized(lon1, lat1, lon2, lat2):
    """
    向量化计算两点间的 Haversine 距离 (km)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 
    return c * r

# --- 2. 消费者函数 (Worker) ---
def qa_worker_process(raw_batch, output_base_path):
    """
    消费者：接收 batch 数据，构建 DataFrame，执行 QA 检查，写入分片文件
    """
    # Stage 2 输出的列名 (根据你之前的 print 内容推断)
    # 注意：user_id index=3, trip_id index=10
    COLUMNS = ["Date", "time_", "time", "user_id", "lon", "lat", "status", "111", "222", "333", "trip_id"]
    
    valid_dfs = []
    
    for raw_trip_data in raw_batch:
        try:
            # 1. 构建 DataFrame (这步在多进程中做，分摊 CPU)
            df = pd.DataFrame(raw_trip_data, columns=COLUMNS)
            
            # --- QA Check 1: 记录数检查 ---
            if len(df) < 5:
                continue

            # 2. 类型转换 (必须转为 float 才能计算)
            # 假设 time 已经是 unix timestamp 字符串
            # lon/lat 是字符串
            lons = df['lon'].astype(float).values
            lats = df['lat'].astype(float).values
            times = df['time'].astype(float).values

            # --- QA Check 2: 时间跳变检查 (Time Gap > 1800s) ---
            # 计算 time_diff = time[i] - time[i-1]
            # np.diff 计算的是后一个减前一个
            time_diffs = np.diff(times) 
            if np.any(time_diffs > 1800):
                # 存在超过 1800s 的间隔，丢弃
                continue

            # --- QA Check 3: 空间跳变检查 (Space Jump > 25km) ---
            # 构造错位数组进行向量化计算
            # arr[:-1] 是前一个点, arr[1:] 是后一个点
            dists = haversine_vectorized(
                lons[:-1], lats[:-1],
                lons[1:], lats[1:]
            )
            
            if np.any(dists > 25):
                # 存在超过 25km 的瞬间跳变，丢弃
                continue

            # === 所有检查通过 ===
            valid_dfs.append(df)

        except Exception as e:
            # print(f"Error in QA: {e}")
            continue

    # --- 批量写入 (无锁模式) ---
    if valid_dfs:
        pid = os.getpid()
        part_file = f"{output_base_path}.part.{pid}"
        
        final_df = pd.concat(valid_dfs)
        # 转 CSV string
        csv_buffer = final_df.to_csv(header=False, index=False)
        
        with open(part_file, 'a') as f:
            f.write(csv_buffer)
            
    return len(raw_batch)

# --- 3. 生产者函数 (基于 main_producer 修改) ---
def stage3_producer(file_path, q, num_processes, batch_size=2000):
    print("\n[Stage 3] Producer started reading file...")
    
    current_trip = None
    current_trip_data = []      
    current_batch_trips = []    
    trip_cnt = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # [关键修改]
            # Stage 2 输出如果是 to_csv(index=False)，通常是逗号分隔
            # 如果你的 Stage 2 输出是用空格分隔，请把这里改成 split(' ')
            parts = line.split(',') 
            
            # 获取 trip_id (假设在最后一列)
            trip_id = parts[-1]
            
            if current_trip is None:
                current_trip = trip_id
                
            if trip_id != current_trip:
                # 把上一条 trip 加入 batch
                current_batch_trips.append(current_trip_data)
                trip_cnt += 1
                
                # Batch 满了发送
                if trip_cnt % batch_size == 0:
                    q.put(current_batch_trips)
                    current_batch_trips = []

                # 重置
                current_trip = trip_id
                current_trip_data = [parts]
            else:
                current_trip_data.append(parts)
                
        # 处理尾部数据
        if current_trip_data:
            current_batch_trips.append(current_trip_data)
            trip_cnt += 1

        if current_batch_trips:
            q.put(current_batch_trips)

    print(f"\n[Stage 3] Producer finished. Total trips: {trip_cnt}")

    # 结束信号
    for _ in range(num_processes):
        q.put(None)

# --- 4. 主函数 Stage 3 ---
def stage3(input_file_path, qa_rslt_file_path):
    stt = time.time()
    print("="*30)
    print(f"Running Stage 3: QA Check")
    print(f"Input: {input_file_path}")
    print(f"Output: {qa_rslt_file_path}")
    
    # 0. 清理旧文件
    if os.path.exists(qa_rslt_file_path):
        os.remove(qa_rslt_file_path)
    # 清理旧的 part 文件
    old_parts = glob.glob(f"{qa_rslt_file_path}.part.*")
    for p in old_parts:
        os.remove(p)

    # 1. 设置并发环境
    num_processes = max(1, multiprocessing.cpu_count() - 2) # 留2个核给 Producer 和系统
    print(f"Using {num_processes} workers.")
    
    manager = Manager()
    q = manager.Queue() # 限制队列大小，防止内存溢出

    pool = Pool(processes=num_processes)
    
    # 2. 启动生产者
    producer_p = multiprocessing.Process(
        target=stage3_producer, 
        args=(input_file_path, q, num_processes)
    )
    producer_p.start()

    # 3. 启动消费者 (主进程循环提交任务)
    pbar = tqdm(desc="QA Progress", unit="batch")
    
    def update_pbar(arg):
        pbar.update(1)

    results = []
    
    while True:
        try:
            batch_data = q.get() # 阻塞获取，直到有数据
        except Exception:
            break

        if batch_data is None:
            break
            
        res = pool.apply_async(
            qa_worker_process,
            args=(batch_data, qa_rslt_file_path),
            callback=update_pbar
        )
        results.append(res)
    
    # 4. 等待完成
    
    pbar.total = len(results)
    pbar.refresh()
    
    pool.close()
    pool.join() # 等所有消费者算完

    pbar.close()

    producer_p.join() # 等生产者读完


    # 5. 合并文件 (Merge)
    print("Merging part files...")
    part_files = glob.glob(f"{qa_rslt_file_path}.part.*")
    
    with open(qa_rslt_file_path, 'wb') as outfile:
        for filename in part_files:
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
            os.remove(filename) # 合并完删除分片

    print(f"Stage 3 Done. Time: {time.time() - stt:.2f}s")

if __name__ == '__main__':

    stage2_output = ""
    stage3_output = ""
    
    stage3(stage2_output, stage3_output)
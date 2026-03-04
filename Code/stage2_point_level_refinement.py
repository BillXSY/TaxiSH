import multiprocessing
from multiprocessing import Pool, Queue, Manager
import pandas as pd
import os
import time
import fcntl
import math
from tqdm import tqdm
import argparse
import numpy as np
import glob

MIN_ANGLE = 20
MAX_SPEED = 150
BATCH_SIZE = 2000
READONLY = False  # 设置为True以启用只读模式，不进行实际写入

file_path = ''
output_file_path = ''

# 用于进程间通信的全局队列（使用Manager创建以支持多进程安全）
global_queue = None


def init_queue(q):
    """初始化全局队列，供工作进程使用"""
    global global_queue
    global_queue = q


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r


def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 
    return c * r


def get_angle_bkp(ref_lon, ref_lat, lon1, lat1, lon2, lat2):
    if lon1 == lon2 and lat1 == lat2:
        return 0.0
    if ref_lon == lon1 and ref_lat == lat1 or ref_lon == lon2 and ref_lat == lat2:
        return 999
    vec1 = (lon1 - ref_lon, lat1 - ref_lat)
    vec2 = (lon2 - ref_lon, lat2 - ref_lat)
    # 计算向量的点积
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    # 计算向量的模
    norm_vec1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    norm_vec2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    try:
        # 计算夹角的余弦值
        cos_angle = round(dot_product / (norm_vec1 * norm_vec2), 5)
        # 计算夹角（弧度）
        angle = math.acos(cos_angle)
        # 将弧度转换为角度
        angle_degrees = math.degrees(angle)
        return angle_degrees
    except Exception as e:
        print(f'{e=}')
        print(ref_lon, ref_lat, lon1, lat1, lon2, lat2)
        print(f'{dot_product=}')
        print(f'{norm_vec1=}')
        print(f'{norm_vec2=}')
        return 0.0


def get_angle_vectorized(ref_lon, ref_lat, lon1, lat1, lon2, lat2):
    """
    向量化计算三点夹角 (Ref为顶点)。
    
    逻辑定义：
    1. P1 == P2 (折返): 视为 0度 (尖角，需过滤)。
    2. Ref == P1 or Ref == P2 (静止/重合): 视为 180度 (安全，交给去重逻辑处理)。
    """
    # 1. 构建向量 (Vector Construction)
    # v1: Ref -> P1
    v1_x = lon1 - ref_lon
    v1_y = lat1 - ref_lat
    
    # v2: Ref -> P2
    v2_x = lon2 - ref_lon
    v2_y = lat2 - ref_lat
    
    # 2. 计算点积 (Dot Product)
    dot_product = v1_x * v2_x + v1_y * v2_y
    
    # 3. 计算模 (Norm)
    norm_v1 = np.sqrt(v1_x**2 + v1_y**2)
    norm_v2 = np.sqrt(v2_x**2 + v2_y**2)
    
    # 4. 计算分母
    denominator = norm_v1 * norm_v2
    
    # === 核心逻辑处理 ===
    
    # 标记 Case B: 分母为0意味着 Ref 与 P1 或 P2 重合
    # is_invalid 表示几何上无法计算角度的点
    is_invalid = (denominator == 0)
    
    # 为了避免除零报错，将分母中的 0 临时替换为 1.0 (计算结果会被稍后覆盖)
    # 使用 np.where(条件, 满足时的值, 不满足时的值)
    safe_denominator = np.where(is_invalid, 1.0, denominator)
    
    # 5. 计算余弦值
    cos_angle = dot_product / safe_denominator
    
    # 6. 数值稳定性截断 (Clip)
    # 浮点数误差可能导致 1.00000001，需限制在 [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 7. 计算角度 (Arccos -> Degrees)
    angle_deg = np.degrees(np.arccos(cos_angle))
    
    # 8. 最终逻辑修正
    # 将 Case B (Ref重合点) 强制设为 180.0 度
    # 这样它们 > MIN_ANGLE，不会被 Sharp Filter 删除，而是留给 Duplicate Filter 处理
    angle_deg = np.where(is_invalid, 180.0, angle_deg)
    
    return angle_deg


def remove_cont_rept_rcrd(data_df):
    if len(data_df) == 0:
        return data_df

    # 如果上游已经保证了一个 trip 一个用户，这行 assert 可以去掉以省时间
    # assert len(data_df['user_id'].unique()) == 1, ...
    
    # 1. 构造“上一行”的数据 (瞬间完成，不需要循环)
    prev_lon = data_df['lon'].shift(1)
    prev_lat = data_df['lat'].shift(1)
    prev_time = data_df['time'].shift(1)
    
    # 2. 向量化计算距离 (瞬间完成)
    # 使用我们之前改好的 vectorized 版本
    dists = haversine_vectorized(data_df['lon'], data_df['lat'], prev_lon, prev_lat)
    
    # 3. 向量化计算时间差
    time_diff = data_df['time'] - prev_time
    
    # 4. 构建布尔掩码 (Boolean Mask)
    # 逻辑：(时间不等) AND (距离 > 0.05)
    # 注意：shift(1) 会导致第一行产生 NaN，需要处理
    
    # 条件 A: 距离 > 0.05 (处理 NaN: 第一行 dist 是 NaN，填 0 或其他，稍后单独强制 True)
    cond_dist = dists.fillna(0) > 0.05
    
    # 条件 B: 时间不同
    cond_time = time_diff.fillna(0) != 0
    
    # 组合条件
    keep_mask = cond_dist & cond_time
    
    # 5. 强制保留第一行
    # 你的原逻辑：if i == 0: keep.append(True)
    keep_mask.iloc[0] = True
    
    # 6. 应用过滤
    return data_df[keep_mask].reset_index(drop=True)


def remove_sharp_rcrd(data_df):
    
    if len(data_df) == 0:
        return data_df
    
    assert len(data_df['user_id'].unique()) == 1, f"remove_sharp_rcrd 数据集中包含多个用户: {data_df['user_id'].unique()}"
    # assert len(data_df['trip_id'].unique()) == 1, '数据集中包含多个分段'

    # 计算、移除夹角小于MIN_ANGLE的记录
    data_df['prev_lon'] = data_df['lon'].shift(1)
    data_df['prev_lat'] = data_df['lat'].shift(1)
    data_df['next_lon'] = data_df['lon'].shift(-1)
    data_df['next_lat'] = data_df['lat'].shift(-1)
    # data_df['next_pos_id'] = data_df['pos_id'].shift(-1)
    # data_df.fillna(0, inplace=True)

    # data_df['angle'] = data_df.apply(lambda x: get_angle(x.lon, x.lat, x.prev_lon, x.prev_lat, x.next_lon, x.next_lat), axis=1)

    data_df['angle'] = get_angle_vectorized(data_df['lon'], data_df['lat'],            # Ref (当前点)
                                            data_df['prev_lon'], data_df['prev_lat'],  # P1 (前一点)
                                            data_df['next_lon'], data_df['next_lat']   # P2 (后一点)
                                            )
    data_df['angle'].fillna(180, inplace=True)

    data_df = data_df.query("angle >= @MIN_ANGLE ")

    return data_df.reset_index(drop=True)


def remove_speeding_rcrd(data_df, max_speed=MAX_SPEED):
    if len(data_df) == 0:
        return data_df
    
    assert len(data_df['user_id'].unique()) == 1, "包含多个用户"

    # 重置索引，确保我们可以通过 idx 访问
    df = data_df.reset_index(drop=True)
    
    # 初始化：第一个点默认保留
    # valid_indices 用于存储保留下来的点的 index
    valid_indices = [0] 

    for i in range(1, len(df)):
        curr_idx = i
        
        # 1. 动态获取“上一个有效点”
        # 注意：这里解决了“多米诺效应”，我们总是和最新的有效点比
        prev_idx = valid_indices[-1]
        
        # 2. 计算速度 (Current vs Prev_Valid)
        dist = haversine(df.loc[curr_idx, 'lon'], df.loc[curr_idx, 'lat'],
                         df.loc[prev_idx, 'lon'], df.loc[prev_idx, 'lat'])
        
        time_diff = df.loc[curr_idx, 'time'] - df.loc[prev_idx, 'time']
        
        # 避免除以0
        if time_diff <= 0:
            # 如果时间相同但距离很大 -> 漂移，不加入
            # 如果时间相同距离为0 -> 重复点，跳过或保留（视需求），这里选择跳过
            continue 
            
        speed = (dist / time_diff) * 3600 # km/h

        # 3. 判断逻辑
        if speed <= max_speed:
            valid_indices.append(curr_idx)
        else:
            # === 触发超速：进入前瞻校验 ===
            # 我们需要判断是 Curr 漂移了，还是 Prev 漂移了
            
            # 如果已经是最后一个点，无法前瞻，只能认为当前点是漂移点，抛弃 Curr
            if i == len(df) - 1:
                continue 
            
            # 获取下一个点 (Next)
            next_idx = i + 1
            next_lon, next_lat = df.loc[next_idx, 'lon'], df.loc[next_idx, 'lat']
            
            # A: Current -> Next 的距离
            dist_curr_next = haversine(df.loc[curr_idx, 'lon'], df.loc[curr_idx, 'lat'],
                                       next_lon, next_lat)
            
            # B: Prev_Valid -> Next 的距离 (Skip Distance)
            dist_prev_next = haversine(df.loc[prev_idx, 'lon'], df.loc[prev_idx, 'lat'],
                                       next_lon, next_lat)
            
            # === 核心博弈逻辑 ===
            # 如果 dist_curr_next >= dist_prev_next:
            # 说明从 Prev 直接去 Next 更近，Current 绕远了。
            # 判决: Current 是噪点。
            # 操作: 不把 curr_idx 加入 valid_indices (即删除 Current)
            if dist_curr_next >= dist_prev_next:
                pass # Drop Current
                
            else:
                # 说明 Current 离 Next 更近，Prev 才是那个导致这种跳变的噪点（例如 Prev 是回跳）
                # 判决: Prev 是噪点，Current 是对的。
                # 操作: 回溯删除 Prev，加入 Current
                valid_indices.pop() # 删除 Prev
                
                # 再次检查：栈空了怎么办？(极罕见，除非由连续噪点引起)
                if not valid_indices:
                    # 如果栈空了，就把当前点当作新的起点
                    valid_indices.append(curr_idx)
                else:
                    # 严谨的做法：这里其实应该再检查一次 Current 和 Prev_Prev 的速度
                    # 但为了简化，我们假设这次修正有效，直接加入
                    valid_indices.append(curr_idx)

    # 4. 根据 index 筛选结果
    return df.loc[valid_indices].reset_index(drop=True)


def trip_process(raw_data_batch, output_file_path, iterations=20):

    if READONLY:
        return len(raw_data_batch)

    COLUMNS = ['Date', 'time_', 'user_id', 'lon', 'lat', 'status', '111', '222', '333', 'trip_id']

    rslt = []

    for raw_trip_data in raw_data_batch:

        try:
            data_df = pd.DataFrame(raw_trip_data, columns=COLUMNS)

            if len(data_df) == 0:
                print(f'EMPTY: trip is empty')
                continue

            trip_id = data_df['trip_id'].unique()[0]

            assert len(data_df['user_id'].unique()) == 1, f"trip_process 数据集中包含多个用户: {data_df['user_id'].unique()}"
            # assert len(data_df['trip_id'].unique()) == 1, '数据集中包含多个分段'

            data_df['lon'] = data_df['lon'].astype(float)
            data_df['lat'] = data_df['lat'].astype(float)

            # data_df['pos_id'] = data_df['lon'].round(4).astype(str) + ',' + data_df['lat'].round(4).astype(str)

            data_df['time'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['time_'], format='%Y-%m-%d %H:%M:%S').astype(int) // 10**9 

            data_df = remove_cont_rept_rcrd(data_df.reset_index(drop=True))

            # print(data_df)

            for iteration in range(iterations):

                if len(data_df) == 0:
                    break

                sharp_num, dup_num, fast_num = 0, 0, 0

                # 移除夹角小于MIN_ANGLE的记录
                prev_len = len(data_df)
                data_df = remove_sharp_rcrd(data_df)
                sharp_num += prev_len - len(data_df)
                # print(f'removed {sharp_num} sharp records in iteration {iteration}')

                # 移除同一地点的连续记录
                prev_len = len(data_df)
                data_df = remove_cont_rept_rcrd(data_df)
                dup_num += prev_len - len(data_df)
                # print(f'removed {dup_num} dup records in iteration {iteration}')

                # 计算、移除速度大于 MAX_SPEED 的记录
                prev_len = len(data_df)
                data_df = remove_speeding_rcrd(data_df)
                fast_num += prev_len - len(data_df)
                # print(f'removed {fast_num} fast records in iteration {iteration}')

                # 移除同一地点的连续记录
                prev_len = len(data_df)
                data_df = remove_cont_rept_rcrd(data_df)
                dup_num += prev_len - len(data_df)
                # print(f'removed {dup_num} dup records in iteration {iteration}')h

                # print(f'Iteration {iteration} for trip {trip_id}: removed {sharp_num} sharp, {dup_num} dup, {fast_num} fast records. Remaining records: {len(data_df)}')

                if sharp_num == 0 and dup_num == 0 and fast_num == 0:
                    break


            if len(data_df) == 0:
                print(f'ALL_REMOVED: All records removed for trip {trip_id}')
                continue


            rslt.append(data_df[["Date", "time_", "time", "user_id", "lon", "lat", "status", "111", "222", "333", "trip_id"]])

        except Exception as e:
            print(f'Error processing trip {trip_id}: get {e}')

    # 优化后的写入部分
    # valid_dfs = [df for df in rslt if len(df) >= 5]

    # 获取当前进程ID
    pid = os.getpid()
    # 每个进程写自己的文件，无需加锁！
    process_output_path = f"{output_file_path}.part.{pid}.csv"


    valid_dfs = rslt

    if valid_dfs:
        # 合并为一个大 DF
        final_batch_df = pd.concat(valid_dfs)
        # 转为 CSV string (在内存中完成，不占锁)
        csv_buffer = final_batch_df[["Date", "time_", "time", "user_id", "lon", "lat", "status", "111", "222", "333", "trip_id"]].to_csv(header=False, index=False)
        
        with open(process_output_path, 'a') as f:
            # fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(csv_buffer) # 只做一次 I/O
            # fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return len(raw_data_batch)
        

def main_producer(file_path, q, num_processes):
    """
    生产者：主进程顺序读取文件，并按trip分组数据，然后将每个trip的数据放入队列
    """
    print("\nProducer started reading file and producing trip data...")
    current_trip = None
    current_trip_data = []
    current_trip_data_batch = []
    trip_cnt = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 假设每行的trip_id在第一个列，以逗号分隔
            parts = line.split(' ')
            trip_id = parts[-1]
            
            if current_trip is None:
                current_trip = trip_id
                
            if trip_id != current_trip:
                # 遇到新的trip，将当前trip的数据放入队列
                # current_trip_data = pd.DataFrame([x.split(' ') for x in current_trip_data], columns=['Date', 'time_', 'user_id', 'lon', 'lat', 'status', '111', '222', '333', 'trip_id'])
                current_trip_data_batch.append(current_trip_data)
                
                trip_cnt += 1
                if trip_cnt % BATCH_SIZE == 0:
                    q.put(current_trip_data_batch)
                    current_trip_data_batch = []

                # 重置当前trip和数据
                current_trip = trip_id
                current_trip_data = [parts]
            else:
                current_trip_data.append(parts)
                
        # 文件读取结束后，放入最后一个trip的数据
        if current_trip_data:
            # current_trip_data = pd.DataFrame([x.split(' ') for x in current_trip_data], columns=['Date', 'time_', 'user_id', 'lon', 'lat', 'status', '111', '222', '333', 'trip_id'])
            current_trip_data_batch.append(current_trip_data)
            trip_cnt += 1

        if current_trip_data_batch:
            q.put(current_trip_data_batch)

    print(f"\nProducer finished. Total trips produced: {trip_cnt}")

    # 发送结束信号：放入None（或其他特定信号）告知消费者不再有数据
    for _ in range(num_processes):
        q.put(None)


def stage2():
    print("stage 2 starts...")

    stt = time.time()

    print(f"{BATCH_SIZE=}")
    print(f"{file_path=}")
    print(f"{output_file_path=}")

    # 如果输出文件已存在，先删除以避免数据混乱
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    old_parts = glob.glob(f"{output_file_path}.part.*")
    for p in old_parts:
        os.remove(p)

    # 设置进程池大小（通常为CPU核心数）
    num_processes = multiprocessing.cpu_count() 
    # num_processes = 32
    print(f'Using {num_processes} processes for parallel processing.')

    
    # 使用Manager创建Queue，支持多进程安全
    with Manager() as manager:
        q = manager.Queue() 
        # q = manager.Queue(maxsize=10000)  # 设置队列大小，避免内存占用过高
        
        # 创建进程池，并初始化全局队列
        with Pool(processes=num_processes, initializer=init_queue, initargs=(q,)) as pool:
            # 启动生产者进程（在主进程执行）
            producer_process = multiprocessing.Process(target=main_producer, args=(file_path, q, num_processes))
            producer_process.start()

            # get pid of the producer process
            print(f'Producer process PID: {producer_process.pid}')

            # 消费者处理：工作进程从队列中取数据并处理
            results = []
            batch_count = 0

            # 创建进度条（如果不知道总数，可以先不设置total参数）
            pbar = tqdm(desc="Processing trips", unit="batch")

            def update_pbar(arg):
                pbar.update(1)

            while True:
                # 从队列中获取trip数据
                try:
                    trip_data_batch = q.get(block=False)
                except:
                    # print(f"Queue is empty, waiting for data...   ", end=' ')
                    trip_data_batch = q.get()
                    # print(f"Got data from queue.")

                # 检查是否为结束信号
                if trip_data_batch is None: 
                    break
                # 使用进程池异步处理每个trip
                result = pool.apply_async(trip_process, (trip_data_batch, output_file_path), callback=update_pbar)
                results.append(result)

                # batch_count += 1
                # if (batch_count * BATCH_SIZE) % 10000 == 0:
                #     pbar.set_description(f"Processing traj {batch_count * BATCH_SIZE}")  # 更新描述
                #     pbar.refresh()  # 立即刷新显示
            
            # pbar.reset(total=len(results))  # 重新设置进度条总数
            pbar.total = len(results)
            pbar.refresh()

            # for i, result in enumerate(results):
            #     result.wait()  # 等待任务完成
            #     pbar.update(1)  # 更新进度条

            pool.close()
            pool.join()

            pbar.close()

            # 等待生产者进程结束
            producer_process.join()

            print("Processing finished. Merging files...")

            # 3. 最后一步：合并所有 part 文件到一个总文件 (如果需要)
            # 这步是顺序 I/O，速度很快
            part_files = glob.glob(f"{output_file_path}.part.*")
            
            with open(output_file_path, 'w') as outfile:
                # 可选：先写个 header
                # outfile.write("Date,time_,...\n") 
                
                for part_file in part_files:
                    with open(part_file, 'r') as infile:
                        # 这种大文件拷贝使用 shutil.copyfileobj 最快
                        import shutil
                        shutil.copyfileobj(infile, outfile)
                    # 合并完删除临时分片
                    os.remove(part_file)
                    
            print("Merge complete.")

            print("All trips processed and written to output file.")



    edt = time.time()
    print(f'Total time: {edt - stt:.2f} seconds')


if __name__ == '__main__':
    stage2()

# 1. 完整 profile（生成 .ncu-rep 报告）
sudo /usr/local/cuda-13.2/bin/ncu \
  --set full \
  --launch-skip 5 \
  --launch-count 1 \
  -f -o ./report_name \
  /home/leo/miniconda3/envs/cutedsl/bin/python your_script.py

# 2. 查看报告摘要（每个 kernel 的关键指标表）
sudo /usr/local/cuda-13.2/bin/ncu -i ./report_name.ncu-rep --print-summary per-kernel

# 3. 查看原始指标（精确数值，可 grep 过滤）
sudo /usr/local/cuda-13.2/bin/ncu -i ./report_name.ncu-rep --page raw | grep "sm__throughput"

# 4. 常用 grep 组合
| grep -E "(gpu__time_duration|launch__registers|launch__block_size|launch__shared_mem|sm__throughput.avg.pct|sm__warps_active.avg.per_cycle|smsp__warps_eligible|l1tex__throughput|dram__throughput)"

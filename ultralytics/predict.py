from ultralytics import YOLO

model = YOLO(r"F:\my_code\yolov12_fuse_SA\package\weights\best.pt")


results = model.predict(source=r"F:\my_code\yolov10_fuse-SA\ultralytics\testimages",
                        imgsz=416,
                        cache='disk',
                        workers=8,
                        device='0',
                        exist_ok=False,
                        save=True,
                        # 是否保存打印特征图
                        visualize=False,
                        name=r".\detect\pre"
                        )


# 统计信息
total_objects = 0
total_time_ms = 0.0

for i, r in enumerate(results, start=1):
    objs = len(r.boxes)
    t = r.speed['preprocess'] + r.speed['inference'] + r.speed['postprocess']
    total_objects += objs
    total_time_ms += t

# 汇总结果
print("\n========== 预测统计结果 ==========")
print(f"总预测时间（模型报告）: {total_time_ms/1000:.3f} 秒")
print(f"平均每张图用时: {(total_time_ms/len(results)):.2f} ms")
print(f"检测到的目标总数: {total_objects} 个 WindTurbine")
print("==================================")



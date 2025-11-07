from ultralytics import YOLO

model = YOLO(r"D:\Git\yolov12_fuse_SA\ultralytics\runs\detect\WindTurbine_6bands_M_500epoch_yolov10_fuse_npy\weights\best.pt")


results = model.predict(source=r"D:\Git\yolov10_fuse\ultralytics\testimages",
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






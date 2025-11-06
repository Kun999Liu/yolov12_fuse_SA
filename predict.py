from ultralytics import YOLO



model = YOLO(r"D:\Git\yolov12_fuse_SA\ultralytics\runs\detect\TransmissionTower_7bands_M_500epoch_yolov12_fuse_npy\weights\best.pt")

results = model.predict(source=r"D:\Git\yolov12_fuse_SA\ultralytics\testdatasets\test\images",
                            imgsz=416,
                            cache='disk',
                            workers=8,
                            device='0',
                            exist_ok=False,
                            save=True,
                            #是否保存打印特征图
                            visualize=False,
                            name=r"D:\Git\yolov12_fuse_SA\run\detect\pre"
                            )
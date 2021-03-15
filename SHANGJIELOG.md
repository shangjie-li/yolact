# SHANGJIELOG

 - 下载YOLACT库
   ```
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   ```

 - 下载权重文件yolact_base_54_800000.pth

 - 安装Pytorch 1.0.1和TorchVision
   ```
   sudo pip3 install torch==1.0.1 -f https://download.pytorch.org/whl/cu90/stable
   sudo pip3 install torchvision==0.2.2
   ```
   
 - 安装cython opencv-python pillow pycocotools matplotlib
   ```
   # Cython needs to be installed before pycocotools.
   pip3 install cython
   pip3 install opencv-python pillow pycocotools matplotlib
   ```

 - 在Ubuntu 16.04中安装nvidia驱动（nvidia Geforce 1050）
   ```
   详情参见https://blog.csdn.net/Scythe666/article/details/84817959
   1.从NVIDIA官网下载对应显卡型号的Linux 64-bit驱动程序NVIDIA-Linux-x86_64-430.34.run并将其保存在Home文件夹（文件目录不要含中文）
   2.卸载旧驱动
   sudo apt-get remove --purge nvidia* # 若安装失败也是这样卸载
   ./NVIDIA-Linux-x86_64-390.48.run --uninstall # 确保卸载干净
   3.安装依赖
   sudo apt-get update 
   sudo apt-get install dkms build-essential linux-headers-generic
   sudo apt-get install gcc-multilib xorg-dev
   sudo apt-get install freeglut3-dev libx11-dev libxmu-dev install libxi-dev  libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
   4.禁用noueau驱动
   sudo vi /etc/modprobe.d/blacklist-nouveau.conf 
   # 在文件blacklist-nouveau.conf中加入如下内容：
   # 开始编辑 i
   blacklist nouveau
   blacklist lbm-nouveau
   options nouveau modeset=0
   alias nouveau off
   alias lbm-nouveau off
   # 退出编辑 Esc 保存 :wq
   # 禁用nouveau内核模块
   echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
   sudo update-initramfs -u
   reboot # 重启计算机
   lsmod |grep nouveau # 无显示则成功
   5.再次重启计算机并进入BIOS中关闭secure boot
   6.按CTRL+ALT+F1进入tty并关闭图形界面（进入全黑界面）
   sudo service lightdm stop
   7.在所保存的目录中运行.run文件并选择合适选项
   chmod a+x NVIDIA-Linux-x86_64-384.90.run # 添加权限
   sudo ./NVIDIA-Linux-x86_64-384.90.run --dkms --no-opengl-files
   8.忽略pre-script failed并继续
   9.安装过程中的选项
   DKMS选yes
   32位兼容选yes
   10.验证
   nvidia-smi # 若列出GPU的信息列表，表示驱动安装成功
   11.重新进入桌面
   sudo service lightdm start # 如果没自动跳转可以按CTRL+ALT+F7
   ```

 - 测试自己的图片和视频
   ```
   python3 eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.jpeg
   python3 eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4
   ```

 - 运行时出现
   ```
   ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type
   报错原因：
   安装的opencv与ros的环境冲突。
   解决方法：
   在需要运行import cv2的coco.py文件中，添加以下代码：
   import sys
   sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
   ```

 - 由于Python版本问题，在运行YOLACT时应将原命令中的python改为python3。

 - 下载COCO数据集，耗时6小时
   ```
   sh data/scripts/COCO.sh # 5000张.jpg图片（814MB）及其标注（12项）
   ```

 - 下载COCO test_dev数据集，耗时2小时
   ```
   sh data/scripts/COCO_test.sh # 40670张.jpg图片（6.6GB）及其标注（image_info_test2017.json和image_info_test-dev2017.json）
   ```

 - 测试Quantitative Results
   ```
   python3 eval.py --trained_model=weights/yolact_base_54_800000.pth
   # 结果：运行时间 15分钟，总量 4952（张），FPS 4.54，box mAP 31.92,mask mAP 29.59
   python3 eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json
   # 结果：运行时间 20分钟，总量 4952（张），FPS 3.77, Dumping bbox_detections.json and mask_detections.json
   python3 run_coco_eval.py
   # 结果：TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.（尚未找到解决方法）
   python3 eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json --dataset=coco2017_testdev_dataset
   # 结果：运行时间 90分钟，总量 20288（张），FPS 3.47, Dumping bbox_detections.json and mask_detections.json
   ```

 - 运行时出现
   ```
   AttributeError: 'NoneType' object has no attribute 'shape'
   报错原因：
   2020年5月1日下载的版本中coco.py文件与较早的版本相比有一些改动。
   解决方法：
   恢复较早版本的coco.py文件可以避免报错。
   ```

 - 测试Qualitative Results
   ```
   python3 eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --display
   # 结果：可以连续地显示单张图片的实例分割效果
   ```

 - 测试Benchmarking
   ```
   python3 eval.py --trained_model=weights/yolact_base_54_800000.pth --benchmark --max_images=1000
   # 结果：运行时间 3分钟，总量 1000（张），FPS 6.88，Time 145.35ms
   ```

 - 制作数据集
   ```
   将KITTI格式数据集的.png图片及其标注的.json文件（labelme生成的）转化为COCO格式数据集的.jpg图片及其标注的.json文件	
   labelme2coco_instance.py
   数据集（存放在./data/kitti/images/中）：1000张train训练集（003000-003999）和100张validation验证集（006900-006999）
   对应的标注文件（存放在./data/kitti/annotations/中）：instances_train_20200505.json和instances_val_20200505.json
   ```
	
 - 用所制作的数据集训练网络
   ```
   config.py中修改如下：
   在dataset_base后：
   my_custom_dataset = dataset_base.copy({
    	'name': 'My Dataset',
    	'train_images': './data/kitti/images/',
    	'train_info':   './data/kitti/annotations/instances_train_20200505.json',
    	'valid_images': './data/kitti/images/',
    	'valid_info':   './data/kitti/annotations/instances_val_20200505.json',
    	'has_gt': True,
    	'class_names': ('vehicle_car', 'vehicle_bus', 'vehicle_van',
			'vehicle_truck', 'vehicle_train', 'human_pedestrian',
			'human_cyclist', 'human_motorcyclist', 'traffic_trafficsign',
			'traffic_trafficlight')
   })
   在yolact_base_config中
   #~ 'dataset': coco2017_dataset,
   #~ 'num_classes': len(coco2017_dataset.class_names) + 1,
   'dataset': my_custom_dataset,
   'num_classes': len(my_custom_dataset.class_names) + 1,
   ```

 - 训练网络（第1次尝试）
   ```
   设置运行程序用的显卡，然后开始训练（在1个GTX1080Ti上）
   CUDA_VISIBLE_DEVICES=1
   python train.py --config=yolact_base_config --batch_size=8
   # 预计训练时间：6天
   # 结果：训练过程中计算机意外重启，更新权重未满10000次，未生成权重文件
   ```

 - 训练网络（第2次尝试）
   ```
   删除logs文件夹，设置运行程序用的显卡，然后开始训练（在2个GTX1080Ti上）
   CUDA_VISIBLE_DEVICES=0,1
   python train.py --config=yolact_base_config --batch_size=16
   # 预计训练时间：3天
   # 平均每次更新权重时间：0.7秒
   # 结果：训练过程中计算机意外重启，更新权重满20000次，生成yolact_base_161_10000.pth和yolact_base_322_20000.pth
   # 以yolact_base_322_20000.pth作为权重运行网络，针对验证集，速度 17 FPS，精度 bbox mAP 14.74 mask mAP 14.37
   ```

 - 训练网络（第3次尝试）
   ```
   删除logs文件夹，从更新权重20000次开始恢复上次训练
   python train.py --config=yolact_base_config --resume=weights/yolact_base_322_20000.pth --start_iter=-1
   # 预计训练时间：3天
   # 平均每次更新权重时间：0.4秒
   # 结果：手动停止，生成yolact_base_184_23002_interrupt.pth
   ```

 - 训练网络（第4次尝试）
   ```
   从更新权重20000次开始恢复上次训练
   python train.py --config=yolact_base_config --resume=weights/yolact_base_322_20000.pth --start_iter=-1 --batch_size=16
   # 预计训练时间：3天
   # 平均每次更新权重时间：0.7秒
   # 每次迭代epoch中，权重更新iteration为70次（所设置的批量随机梯度下降中的批量大小batch_size为16，而70*16近似为样本总量）
   # 每2次迭代，终端中会给出针对验证集的mAP
   # 结果：手动停止，生成yolact_base_1799_111556_interrupt.pth
   ```

 - 训练网络（第5次尝试）
   ```
   从更新权重111556次开始恢复上次训练
   python train.py --config=yolact_base_config --resume=weights/yolact_base_1799_111556_interrupt.pth --start_iter=-1 --batch_size=16
   # 预计训练时间：2天
   # 平均每次更新权重时间：0.7秒
   # 为加快训练过程，设置每30次迭代计算验证集mAP
   # 结果：训练结束，共更新权重400000次，生成yolact_base_6451_400000.pth
   # 最终，针对验证集，速度 18 FPS，精度 bbox mAP 20.57 mask mAP 18.05
   # 然而，针对训练集，精度可达50 mAP以上（过拟合）
   ```

 - 针对一个全新的test测试数据集，评判所训练的网络性能（FPS、bbox mAP、mask mAP）的方法
   ```
   将用于测试的KITTI格式数据集的.png图片及其标注的.json文件（labelme生成的）转化为COCO格式数据集的.jpg图片及其标注的.json文件
   labelme2coco_instance.py
   数据集（存放在./data/kitti/images_test/中）：150张test测试集（000000-000149）
   对应的标注文件（存放在./data/kitti/annotations_test/中）：instances_test_20200505.json
   config.py中修改如下：
   在dataset_base后：
   my_custom_dataset = dataset_base.copy({
    	'name': 'My Dataset',
    	'train_images': './data/kitti/images/',
    	'train_info':   './data/kitti/annotations/instances_train_20200505.json',
		'valid_images': './data/kitti/images_test/',
		'valid_info':   './data/kitti/annotations_test/instances_test_20200505.json',
    	'has_gt': True,
    	'class_names': ('vehicle_car', 'vehicle_bus', 'vehicle_van',
			'vehicle_truck', 'vehicle_train', 'human_pedestrian',
			'human_cyclist', 'human_motorcyclist', 'traffic_trafficsign',
			'traffic_trafficlight')
   })
   测试：
   python eval.py --trained_model=weights/yolact_base_6451_400000.pth
   结果：bbox mAP 15.70 mask mAP 11.51（多次测试的结果都很稳定）
   ```

 - 运行时出现
   ```
   File "eval.py", line 390, in prep_metrics
		gt_boxes = torch.Tensor(gt[:, :4])
   TypeError: 'NoneType' object is not subscriptable
   报错原因：
   尚未确定原因，猜测是由于数据集标注格式错误（004000-004999）。
   解决方法：
   更换数据集，再次尝试。
   ```









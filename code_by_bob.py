# coding=utf8
import os
import numpy as np
from glob import glob
import shutil
from matplotlib import pyplot as plt

# ===================== 1. 路径配置（保持和你当前目录一致） =====================
train_dir = "train"
test_dir = "test"

# 检查路径
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"未找到 {train_dir} 文件夹！")


# ===================== 2. 快速划分训练/验证集+分类文件夹 =====================
# 读取猫/狗图片（复用之前的逻辑）
cat_imgs = glob(os.path.join(train_dir, "cat.*.jpg"))
dog_imgs = glob(os.path.join(train_dir, "dog.*.jpg"))
print(f"找到猫图：{len(cat_imgs)} 张 | 狗图：{len(dog_imgs)} 张")

# 快速划分（8:2）
np.random.seed(42)
np.random.shuffle(cat_imgs)
np.random.shuffle(dog_imgs)
val_cat_num = int(len(cat_imgs)*0.2)
val_dog_num = int(len(dog_imgs)*0.2)
train_cats, val_cats = cat_imgs[val_cat_num:], cat_imgs[:val_cat_num]
train_dogs, val_dogs = dog_imgs[val_dog_num:], dog_imgs[:val_dog_num]

# 创建分类文件夹（复用）
classify_dir = "train_val_classified"
os.makedirs(os.path.join(classify_dir, "train", "cats"), exist_ok=True)
os.makedirs(os.path.join(classify_dir, "train", "dogs"), exist_ok=True)
os.makedirs(os.path.join(classify_dir, "val", "cats"), exist_ok=True)
os.makedirs(os.path.join(classify_dir, "val", "dogs"), exist_ok=True)

# 快速复制（用shutil.copytree简化，比循环快）
def copy_imgs(imgs, target):
    for img in imgs:
        shutil.copy(img, target)
copy_imgs(train_cats, os.path.join(classify_dir, "train", "cats"))
copy_imgs(train_dogs, os.path.join(classify_dir, "train", "dogs"))
copy_imgs(val_cats, os.path.join(classify_dir, "val", "cats"))
copy_imgs(val_dogs, os.path.join(classify_dir, "val", "dogs"))


# ===================== 3. 启用GPU加速（关键！） =====================
import tensorflow as tf
# 检查GPU是否可用
print("\n=== GPU状态检查 ===")
print(f"GPU是否可用：{tf.test.is_gpu_available()}")
print(f"可用GPU列表：{tf.config.list_physical_devices('GPU')}")
# 强制使用GPU（如果有）
if tf.test.is_gpu_available():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("⚠️  未检测到GPU，训练会较慢！建议用N卡并安装CUDA")


# ===================== 4. 轻量数据生成器（简化增强） =====================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # 对应MobileNetV2的预处理

# 简化数据增强（只保留必要的，减少计算）
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True  # 只保留水平翻转，减少计算
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 生成器（增大批次到64，提速）
train_gen = train_datagen.flow_from_directory(
    os.path.join(classify_dir, "train"),
    target_size=(224, 224),
    batch_size=64,  # 批次增大到64（GPU内存够的话）
    class_mode="binary"
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(classify_dir, "val"),
    target_size=(224, 224),
    batch_size=64,
    class_mode="binary"
)
print(f"\n训练集样本数：{train_gen.samples} | 验证集样本数：{val_gen.samples}")


# ===================== 5. 换轻量模型（MobileNetV2） =====================
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # 用GlobalAveragePooling替代Flatten（更轻量）
from tensorflow.keras.optimizers import Adam

# 加载MobileNetV2（轻量、速度快）
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # 先冻结


# 构建轻量模型
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # 比Flatten参数量少很多
    Dense(128, activation="relu"),  # 减少全连接层参数量
    Dense(1, activation="sigmoid")
])


# 编译（用Adam优化器，收敛更快）
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["acc"]
)
model.summary()  # 可以看到模型参数量比VGG16少很多


# ===================== 6. 快速训练（减少轮数） =====================
print("\n=== 开始快速训练 ===")
history = model.fit(
    train_gen,
    epochs=5,  # 只训练5轮（MobileNetV2收敛快）
    validation_data=val_gen,
    steps_per_epoch=train_gen.samples//64,
    validation_steps=val_gen.samples//64
)


# ===================== 7. 绘制曲线 =====================
def plot_history(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(epochs, acc, "r-", label="训练准确率")
    plt.plot(epochs, val_acc, "b-", label="验证准确率")
    plt.legend()
    plt.subplot(122)
    plt.plot(epochs, loss, "r-", label="训练损失")
    plt.plot(epochs, val_loss, "b-", label="验证损失")
    plt.legend()
    plt.show()

plot_history(history)


# ===================== 8. 保存模型 =====================
model.save("cats_dogs_mobilenet_model.h5")
print("\n模型已保存，训练完成！")

import os
import matplotlib.pyplot as plt
import tensorflow as tf

# train데이터와 validation 데이터 경로지정(파일 잘 불러오는지 확인차)
train_dir = './train' 
validation_dir = './validation'

# train 파일 이름 리스트(훈련용 사진)
train_names = os.listdir(train_dir)
print(train_names[:10])

# validation 파일 이름 리스트(정확성 테스트 확인 위한 사진)
validation_names = os.listdir(validation_dir)
print(train_names[:10])

# 총 이미지 파일 개수
print('training images folder:', len(os.listdir(train_dir)))
print('validation images folder:', len(os.listdir(validation_dir)))
#여기까진 파일 잘 불러왔나 확인용



# 모델 구성

model = tf.keras.models.Sequential([ #Sequential클래스 사용하여 인공신경망의 각 층 순서대로 쌓기
# Conv2D(합성곱)층
#첫번째 인자 16 = filters값 (합성곱 연산에서 사용되는 필터는 이미지에서 특징(feature)을 분리해내는 기능을 합니다.
#filters의 값은 합성곱에 사용되는 필터의 종류 (개수)이며, 출력 공간의 차원 (깊이)을 결정합니다.
#두번째 인자 (3,3) = kernel_size값 (kernel_size는 합성곱에 사용되는 필터(=커널)의 크기입니다.
#3×3 크기의 필터가 사용되며, 합성곱 연산이 이루어지고 나면 이미지는 (28, 28) 크기에서 (26, 26)이 됩니다.
#세번째 인자 activation='relu' = 활성화함수 (Activation function)는 ‘relu’로 지정
#마지막 인자 input_shape=(300,300,3) = 입력 데이터의 형태 (input_shape)는 이미지 하나의 형태에 해당하는 (300, 30, 3)로 설정합니다.

#풀링 (Pooling)은 합성곱에 의해 얻어진 Feature map으로부터 값을 샘플링해서 정보를 압축하는 과정을 의미합니다.
#MaxPool2D(2,2) = 특정 영역에서 가장 큰 값을 샘플링하는 풀링 방식, 풀링 필터의 크기를 2X2 영역으로 설정
#이러한 합성곱, 풀링 층은 특성 추출 (Feature Extraction)을 담당하며, 전체 합성곱 신경망의 앞부분을 구성합니다.

    # The first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    
#Flatten층은 2차원 배열의 이미지 포맷을 1차원 배열로 변환하여 평탄화
# Dense(분류담당)층
# 첫번째 인자 = 노드(또는 뉴런) 갯수
# 마지막 층 활성화 함수 sigmoid(데이터 분류위한)
    # Flatten
    tf.keras.layers.Flatten(),
    # 512 Neuron (Hidden layer)
    tf.keras.layers.Dense(512, activation='relu'),
    # 분류 클래스 두개일때 sigmoid, 그 이상 softmax
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary() #구성한 신경망에 대한 정보 출력

# 모델 컴파일
# compile() 메서드를 이용해서 손실 함수 (loss function)와 옵티마이저 (optimizer)를 지정
# 손실 함수로 ‘binary_crossentropy’를 사용
# 옵티마이저 RMSprop
# RMSprop (Root Mean Square Propagation) Algorithm은 훈련 과정 중에 학습률을 적절하게 변화
opt = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
model.compile(loss='binary_crossentropy',#클래스 2개일때 binary_crossentropy
            optimizer=opt,               #그 이상이면 categorical_crossentropy
            metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator 객체의 rescale 파라미터를 이용해서 모든 데이터를 255로 나누어준 다음
#(0~255사이의 값을 갖는 데이터를 0~1사이의 값을 갖도록 변경),
# flow_from_directory() 메소드를 이용해서 훈련과 테스트에 사용될 이미지 데이터를 만듭니다.
# 첫번째 인자로 이미지들이 위치한 경로를 입력하고, batch_size, class_mode 를 지정
# target_size에 맞춰서 이미지의 크기 조절
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
  './train',#훈련용 오토바이, 자동차 폴더 상위폴더 경로
  batch_size=80,
  class_mode='binary', #이진라벨 반환
  target_size=(150, 150)
)

test_generator = test_datagen.flow_from_directory(
  './validation',#확인(테스트)용 오토바이, 자동차 폴더 상위폴더 경로
  batch_size=6,
  class_mode='binary',
  target_size=(150, 150)
)


# 앞에 구성한 Neural Network 모델을 훈련
# 훈련과 테스트를 위한 train_generator, validation_data를 입력
# epochs는 데이터셋을 한번 훈련하는 과정을 의미
# steps_per_epoch는 한 번의 에포크 (epoch)에서 훈련에 사용할 배치 (batch)의 개수를 지정
# validation_steps는 한 번의 에포크가 끝날 때, 테스트에 사용되는 배치 (batch)의 개수를 지정

history = model.fit(
  train_generator,
  validation_data=test_generator,
  steps_per_epoch=100, #data갯수/batch_size
  epochs=10,
  validation_steps=10, #data갯수/batch_size
  verbose=2,
)
#validation 테스트 정확도 상대적으로 낮은이유:  과적합 (Overfitting)

# 정확도와 손실 확인
acc = history.history['accuracy'] # 매 에포크 마다의 훈련 정확도
val_acc = history.history['val_accuracy'] # 매 에포크 마다의 검증 정확도
loss = history.history['loss'] # 매 에포크 마다의 훈련 손실값
val_loss = history.history['val_loss'] # 매 에포크 마다의 검증 손실값

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy') # 파란점
plt.plot(epochs, val_acc, 'b', label='Validation accuracy') # 파란선
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()



#test데이터 경로
test_dir = './test/test_img/'


import glob
import math
import numpy as np
from keras.preprocessing import image

#test데이터 경로의 모든 jpg 파일의 경로 리스트
images_path = glob.glob(test_dir+'*.jpg')


#test데이터들을 target_size로 배열 img에 저장
img = []
for path in images_path:
    img.append(image.load_img(path, target_size=(150,150)))

#배열 img에 저장된 이미지들을 배열 images에 NumPy 배열로 변환하여 저장
images = []
for i in img:
    x = image.img_to_array(i)
    x = np.expand_dims(x, axis=0)
    images.append(np.vstack([x]))

#훈련된 모델로 예측한 이미지들의 클래스를 배열 classes에 저장
classes = []
for i in images:
    classes.append(model.predict(i, batch_size = 10)[0])

#결과 창 하나에 보여지는 이미지 개수와 한 줄에 보여질 이미지의 개수를 지정
nplot_img = 32
nplot_columns = 8

#결과 창 개수와 행의 개수
nplot_subs = math.ceil(len(img)/nplot_img)
nplot_rows = math.ceil(nplot_img/nplot_columns)

#이미지와 예측한 클래스를 타이틀로 출력하는 함수 정의
#row, column의 서브플롯에 position 번째에 이미지를 출력
def plot_subplot(row, column, position, index):
    plt.subplot(row, column, position)
    plt.imshow(img[index])
    if classes[index]>0:
        title = "no motorcycle"
    else:
        title = "motorcycle"
    plt.gca().set_title(title)
    plt.axis("off")

    

#결과 창에 이미지 나열
for i in range(nplot_subs):
    plt.figure(figsize=(200, 150))
    if i ==  (nplot_subs-1):
        for j in range(i * nplot_img, len(img)):
            plot_subplot(nplot_rows, nplot_columns,j+1-(i * nplot_img), j)            
    else:
        for j in range(i * nplot_img, (i+1) * nplot_img):
            plot_subplot(nplot_rows, nplot_columns,j+1-(i * nplot_img), j)


plt.show()








